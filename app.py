import io
import os
import re
import json
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from pypdf import PdfReader
from openai import OpenAI

# ===================== CONFIG =====================
st.set_page_config(page_title="EPIC NL Search — LLM PoC", page_icon="🩺", layout="wide")
st.title("🩺 EPIC NL Search — LLM-first PoC")

st.markdown("""
This PoC uses a **Large Language Model** to:
1) **Extract** structured fields from uploaded clinical PDFs (pathology/radiology): TNM stage, tumor size (cm), key dates, recurrence/progression (+ negation), cancer type.
2) **Parse your natural question** into filters and return a cohort table you can download as CSV.

**Notes**
- Use **de-identified, text-based PDFs** for now (no OCR).
- Set your **OPENAI_API_KEY** as an environment variable in your deployment (Streamlit Cloud → Secrets, or locally via shell).
""")

# ===================== OPENAI CLIENT =====================
def get_openai_client() -> Optional[OpenAI]:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        st.error("OPENAI_API_KEY is not set. Please configure it in your environment/secrets to use the LLM.")
        return None
    try:
        client = OpenAI()
        return client
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {e}")
        return None

# Choose model (fast+cheap by default)
with st.sidebar:
    st.header("⚙️ Settings")
    model = st.selectbox("OpenAI model", ["gpt-4o-mini", "gpt-4o"], index=0, help="Use 4o-mini for speed/cost; 4o for maximum quality.")
    max_chars = st.slider("Max characters per document sent to LLM", min_value=2000, max_value=16000, step=1000, value=7000)
    st.caption("Large PDFs are truncated to this length to control cost.")

# ===================== DATA MODEL =====================
@dataclass
class PatientFacts:
    patient_id: str
    age: Optional[int] = None
    bmi: Optional[float] = None
    cancer_type: Optional[str] = None
    tnm_stage: Optional[str] = None
    tumor_size_cm: Optional[float] = None
    diagnosis_date: Optional[str] = None
    surgery_date: Optional[str] = None
    treatments: List[Dict] = None
    recurrence: Optional[Dict] = None  # {"has_recurrence": bool, "date": str|None, "site": str|None}
    last_ct_date: Optional[str] = None
    death_date: Optional[str] = None
    sources: List[Dict] = None

def ensure_state():
    if "facts_map" not in st.session_state:
        st.session_state.facts_map: Dict[str, PatientFacts] = {}
    if "facts" not in st.session_state:
        st.session_state.facts = pd.DataFrame()

ensure_state()

# ===================== UTILITIES =====================
def pdf_to_text(file_bytes: bytes) -> str:
    """Extract text from a PDF (no OCR)."""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text = "".join([(p.extract_text() or "") for p in reader.pages])
        return text
    except Exception:
        return ""

def guess_kind_from_name(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ["ct", "mr", "mri", "pet", "scan", "radiology", "imaging", "chest", "abd", "pelvis"]):
        return "radiology"
    if any(k in n for k in ["path", "surg", "biopsy", "histology", "specimen", "gross", "microscopic", "turbt"]):
        return "pathology"
    return "pathology"

def derive_pid_from_name(name: str) -> str:
    # leading token of filename (without extension) is treated as patient ID
    stem = name.rsplit(".", 1)[0]
    tok = re.split(r'[\s\-_]+', stem)
    return tok[0] if tok else stem

def facts_map_to_df():
    df = pd.DataFrame([asdict(v) for v in st.session_state.facts_map.values()])
    if "recurrence" in df.columns:
        df["recurrence_flag"] = df["recurrence"].apply(lambda x: bool(isinstance(x, dict) and x.get("has_recurrence")))
        df["recurrence_date"] = df["recurrence"].apply(lambda x: x.get("date") if isinstance(x, dict) else None)
        df["recurrence_site"] = df["recurrence"].apply(lambda x: x.get("site") if isinstance(x, dict) else None)
    st.session_state.facts = df

def safe_json_loads(s: str) -> dict:
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r'\{.*\}', s, flags=re.S)
        return json.loads(m.group(0)) if m else {}

# ===================== LLM PROMPTS =====================
EXTRACT_SYSTEM = """You are a clinical information extraction service.
Return a SINGLE JSON object only — no extra text — with the exact schema given."""

EXTRACT_USER_TMPL = """Extract structured fields from the clinical {kind} report text below.
Use this JSON schema exactly; unknowns must be null.

{{
  "patient_id": "{pid}",
  "cancer_type": null,
  "tnm_stage": null,
  "tumor_size_cm": null,
  "diagnosis_date": null,
  "surgery_date": null,
  "recurrence": {{
    "has_recurrence": false,
    "date": null,
    "site": null
  }}
}}

Guidelines:
- cancer_type: "Bladder cancer" if text indicates bladder/urothelial primary; else null.
- tnm_stage: normalized like "pT3a N0 M0" (or "T3 N1 Mx" if pathologic marker absent).
- tumor_size_cm: numeric in **cm** if present.
- diagnosis_date/surgery_date: ISO-like dates if present (YYYY-MM-DD if you can infer, else as written).
- recurrence: True only if there is **positive evidence** of RECURRENCE/PROGRESSION/METASTASIS; set site if described
  (e.g., "Lymph nodes", "Bladder/Pelvis (GU)", "Distant (lung/chest)", "Distant (other)"). If the text says **no** evidence of
  recurrence/metastasis/progression/lymphadenopathy, keep has_recurrence = false.

TEXT:
<<<
{body}
>>>
"""

PARSE_SYSTEM = """You convert natural-language cohort questions into compact filter JSON. Return a SINGLE JSON object, no extra text."""

PARSE_USER_TMPL = """Turn this question into filters.

QUESTION:
{q}

Return JSON with this schema:
{
  "filters": {
    "cancer_type": "bladder" | null,
    "tnm_include": [ "PT3", "PT4" ],   // normalized TNM T-specs to include (T3/T4/pT3/pT4 -> "PT3"/"PT4")
    "recurrence_required": true | false,
    "size_cm": { "op": ">=" | "<" | "<=" | ">" | "==", "value": number } | null
  }
}
Rules:
- If the question asks "WITH recurrence/progression", set recurrence_required=true. If it merely says "include recurrence" as a column, set false.
- Normalize T3/T4 or pT3/pT4 to "PT3"/"PT4" in tnm_include.
- If a cm threshold is given (like ">= 3 cm" or "size > 3 cm" or "3 cm"), set size_cm accordingly (default operator is ">=" if only a number is given).
- If the question says bladder cancer (e.g., "bladder"), set cancer_type="bladder".
"""

# ===================== LLM CALLS =====================
def llm_extract_struct(client: OpenAI, model: str, body: str, pid: str, kind: str) -> dict:
    body = (body or "")[:max_chars]  # truncate long docs for cost control
    messages = [
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user", "content": EXTRACT_USER_TMPL.format(kind=kind, pid=pid, body=body)}
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"}
    )
    return safe_json_loads(resp.choices[0].message.content)

def llm_parse_query(client: OpenAI, model: str, q: str) -> dict:
    messages = [
        {"role": "system", "content": PARSE_SYSTEM},
        {"role": "user", "content": PARSE_USER_TMPL.format(q=q)}
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"}
    )
    return safe_json_loads(resp.choices[0].message.content)

# ===================== SIDEBAR: UPLOADS =====================
with st.sidebar:
    st.subheader("📥 Upload de-identified PDFs")
    uploads = st.file_uploader(
        "Filenames should include a patient ID",
        type=["pdf"],
        accept_multiple_files=True
    )
    col1, col2 = st.columns(2)
    with col1:
        ingest_btn = st.button("Process uploads", use_container_width=True)
    with col2:
        clear_btn = st.button("Clear all", use_container_width=True)

if clear_btn:
    st.session_state.facts_map.clear()
    st.session_state.facts = pd.DataFrame()
    st.success("Cleared all in-memory data.")

# ===================== INGEST (LLM-EXTRACT) =====================
if ingest_btn:
    client = get_openai_client()
    if client is None:
        st.stop()

    if not uploads:
        st.warning("No PDFs selected.")
    else:
        added = 0
        with st.spinner("Extracting with LLM..."):
            for up in uploads:
                raw = up.read()
                text = pdf_to_text(raw)
                if not text.strip():
                    st.warning(f"Skipped (no extractable text): {up.name}")
                    continue

                pid = derive_pid_from_name(up.name)
                kind = guess_kind_from_name(up.name)  # 'pathology' or 'radiology'
                j = llm_extract_struct(client, model, text, pid=pid, kind=kind)

                pf = st.session_state.facts_map.get(pid, PatientFacts(patient_id=pid, treatments=[], sources=[]))

                # Merge LLM output into PatientFacts
                pf.cancer_type   = j.get("cancer_type") or pf.cancer_type
                pf.tnm_stage     = j.get("tnm_stage") or pf.tnm_stage
                val = j.get("tumor_size_cm"); 
                if isinstance(val, (int, float)): 
                    if pf.tumor_size_cm is None or float(val) > float(pf.tumor_size_cm):
                        pf.tumor_size_cm = float(val)
                if j.get("diagnosis_date"): pf.diagnosis_date = j["diagnosis_date"]
                if j.get("surgery_date"):   pf.surgery_date   = j["surgery_date"]

                rec = j.get("recurrence") or {}
                if isinstance(rec, dict) and rec.get("has_recurrence") is True:
                    pf.recurrence = {
                        "has_recurrence": True,
                        "date": rec.get("date"),
                        "site": rec.get("site"),
                    }
                    # last_ct_date best-effort from recurrence date if present
                    if not pf.last_ct_date and rec.get("date"):
                        pf.last_ct_date = rec["date"]

                if pf.sources is None: pf.sources = []
                pf.sources.append({"type": f"{kind}_report", "origin": "upload", "name": up.name})

                st.session_state.facts_map[pid] = pf
                added += 1

        facts_map_to_df()
        st.success(f"Processed {added} PDFs. Patients in memory: {len(st.session_state.facts)}")

# ===================== OPTIONAL DEMO BUTTON (LLM) =====================
with st.sidebar:
    st.subheader("⚡ Demo (no files)")
    demo_btn = st.button("Add demo patients (BLD123, BLD999) via LLM")

if demo_btn:
    client = get_openai_client()
    if client is None:
        st.stop()

    demo_docs = [
        ("BLD123_pathology.txt", "pathology", "BLD123",
         "Final Diagnosis: Invasive urothelial carcinoma (bladder). Pathologic stage: pT3a N0 M0. Tumor size 5.2 cm. Date: 2024-06-14."),
        ("BLD123_ct.txt", "radiology", "BLD123",
         "CT CHEST 2024-10-21. New lesion and mediastinal lymphadenopathy. Impression: progression consistent with recurrence."),
        ("BLD999_pathology.txt", "pathology", "BLD999",
         "Urothelial carcinoma of the bladder. Pathologic stage: pT4 N1 M0. Tumor size 3.2 cm. Date: 2024-05-03."),
        ("BLD999_ct.txt", "radiology", "BLD999",
         "CT 2025-01-05. No lymphadenopathy. No evidence of recurrence or metastasis.")
    ]
    with st.spinner("Extracting demo with LLM..."):
        for name, kind, pid, text in demo_docs:
            j = llm_extract_struct(get_openai_client(), model, text, pid=pid, kind=kind)
            pf = st.session_state.facts_map.get(pid, PatientFacts(patient_id=pid, treatments=[], sources=[]))
            pf.cancer_type   = j.get("cancer_type") or pf.cancer_type
            pf.tnm_stage     = j.get("tnm_stage") or pf.tnm_stage
            val = j.get("tumor_size_cm"); 
            if isinstance(val, (int, float)): 
                if pf.tumor_size_cm is None or float(val) > float(pf.tumor_size_cm):
                    pf.tumor_size_cm = float(val)
            if j.get("diagnosis_date"): pf.diagnosis_date = j["diagnosis_date"]
            if j.get("surgery_date"):   pf.surgery_date   = j["surgery_date"]
            rec = j.get("recurrence") or {}
            if isinstance(rec, dict) and rec.get("has_recurrence") is True:
                pf.recurrence = {"has_recurrence": True, "date": rec.get("date"), "site": rec.get("site")}
                if not pf.last_ct_date and rec.get("date"):
                    pf.last_ct_date = rec["date"]
            if pf.sources is None: pf.sources = []
            pf.sources.append({"type": f"{kind}_report", "origin": "demo", "name": name})
            st.session_state.facts_map[pid] = pf

    facts_map_to_df()
    st.success("Added demo patients via LLM.")

# ===================== LLM QUERY PARSING + FILTERING =====================
def apply_filters(df: pd.DataFrame, fjson: dict) -> pd.DataFrame:
    out = df.copy()
    f = (fjson or {}).get("filters", {})

    # cancer type
    if f.get("cancer_type") == "bladder" and "cancer_type" in out.columns:
        out = out[out["cancer_type"].astype(str).str.contains("bladder", case=False, na=False)]

    # TNM include (list like ["PT3","PT4"])
    tnm_list = f.get("tnm_include") or []
    if tnm_list and "tnm_stage" in out.columns:
        mask = pd.Series(False, index=out.index)
        for t in tnm_list:
            rx = r'\b' + re.escape(t) + r'\b'
            mask = mask | out["tnm_stage"].astype(str).str.upper().str.contains(rx, regex=True, na=False)
        out = out[mask]

    # recurrence
    if f.get("recurrence_required") and "recurrence" in out.columns:
        out = out[out["recurrence"].apply(lambda x: bool(isinstance(x, dict) and x.get("has_recurrence")))]

    # size
    size_spec = f.get("size_cm")
    if size_spec and "tumor_size_cm" in out.columns:
        op = size_spec.get("op", ">=")
        try:
            val = float(size_spec.get("value", 0))
        except Exception:
            val = None
        if val is not None:
            size = pd.to_numeric(out["tumor_size_cm"], errors="coerce")
            if op == ">": out = out[size > val]
            elif op == ">=": out = out[size >= val]
            elif op == "<": out = out[size < val]
            elif op == "<=": out = out[size <= val]
            elif op == "==": out = out[size == val]

    # pretty columns
    cols = ["patient_id","age","bmi","tnm_stage","tumor_size_cm",
            "diagnosis_date","surgery_date","last_ct_date","death_date",
            "cancer_type","recurrence","recurrence_flag","recurrence_date","recurrence_site","sources"]
    cols = [c for c in cols if c in out.columns]
    return out.loc[:, cols]

# ===================== MAIN QUERY UI =====================
st.markdown("### 🔎 Ask a question (LLM parsed)")
q_default = "All bladder cancer patients with stage T3 or T4; include age, TNM, tumor size"
question = st.text_input("Natural-language query", value=q_default)

colA, colB = st.columns([1,1])
with colA:
    run_btn = st.button("Run query with LLM", type="primary")
with colB:
    export_all_btn = st.button("Download full dataset CSV")

if run_btn:
    if st.session_state.facts.empty:
        st.warning("No data yet. Upload PDFs or use the demo button in the sidebar.")
    else:
        client = get_openai_client()
        if client is None:
            st.stop()
        with st.spinner("Parsing question with LLM..."):
            parsed = llm_parse_query(client, model, question)
        st.caption(f"Parsed filters: `{json.dumps(parsed, ensure_ascii=False)}`")
        res = apply_filters(st.session_state.facts, parsed)
        st.success(f"{len(res)} rows")
        st.dataframe(res, use_container_width=True, height=420)
        st.download_button("Download results CSV", data=res.to_csv(index=False).encode("utf-8"),
                           file_name="cohort.csv", mime="text/csv")

if export_all_btn:
    if st.session_state.facts.empty:
        st.warning("No data to export yet.")
    else:
        st.download_button("Download full dataset CSV",
                           data=st.session_state.facts.to_csv(index=False).encode("utf-8"),
                           file_name="patient_facts_full.csv", mime="text/csv")

st.markdown("""
#### Example queries
- **All bladder cancer patients with stage T3 or T4; include age, TNM, tumor size**
- **Bladder cancer patients WITH recurrence or progression; show site and last CT date**
- **Bladder cancer with stage T3 and size >= 3 cm**
- **Patients WITH recurrence or progression; show site and last CT date** (no bladder filter)
""")
