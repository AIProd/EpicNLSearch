import io
import os
import re
import json
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import streamlit as st
from pypdf import PdfReader
from openai import OpenAI

# ===================== CONFIG =====================
st.set_page_config(page_title="EPIC NL Search — LLM PoC", page_icon="🩺", layout="wide")
st.title("🩺 EPIC NL Search — LLM-first PoC")

st.markdown("""
Upload **de-identified, text-based PDFs**. This LLM-first app will:
- Extract patient ID (MRN or name+DOB), dates, TNM/tumor size (if present),
- Detect **metastasis sites** (liver/lung/nodes/bone/other), **nodal disease**, and **recurrence/progression** when appropriate,
- Merge multiple reports for the same patient,
- Parse your natural-language question and return a cohort table you can download.

**Note**: “Initial staging” PET/CT with metastatic disease is recorded as **metastasis**, not **recurrence**.
""")

# ===================== OPENAI CLIENT =====================
def get_openai_client() -> Optional[OpenAI]:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        st.error("OPENAI_API_KEY is not set. Configure it in your environment or Streamlit Secrets.")
        return None
    try:
        client = OpenAI()
        return client
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {e}")
        return None

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    model = st.selectbox(
        "OpenAI model",
        ["gpt-4o-mini", "gpt-4o"],
        index=0,
        help="Use 4o-mini for speed/cost; 4o for max quality."
    )
    max_chars = st.slider("Max characters per document sent to LLM",
                          min_value=2000, max_value=20000, step=1000, value=9000)
    st.caption("Large PDFs are truncated to this length to control cost.")

# ===================== DATA MODEL =====================
@dataclass
class PatientFacts:
    patient_id: str                         # canonical key (MRN preferred)
    mrn: Optional[str] = None
    name: Optional[str] = None
    dob: Optional[str] = None

    age: Optional[int] = None
    bmi: Optional[float] = None
    cancer_type: Optional[str] = None       # e.g., "Bladder cancer", "Renal cell carcinoma"
    tnm_stage: Optional[str] = None
    tumor_size_cm: Optional[float] = None
    diagnosis_date: Optional[str] = None
    surgery_date: Optional[str] = None

    # Disease status
    recurrence: Optional[Dict] = None       # {"has_recurrence": bool, "date": str|None, "site": str|None, "reason": str|None}
    last_ct_date: Optional[str] = None
    death_date: Optional[str] = None
    nodal_disease: Optional[bool] = None    # any mention of nodal disease
    metastasis: Optional[Dict] = None       # {"has_metastasis": bool, "sites": [...]}

    # 👇 Added to match instantiation elsewhere
    treatments: Optional[List[Dict]] = None

    sources: List[Dict] = None              # [{"type": "...", "origin": "upload/demo", "name": filename}]

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
    if any(k in n for k in ["pet", "pet/ct", "nm", "nuc", "nuclear"]):
        return "radiology_petct"
    if any(k in n for k in ["ct", "cta", "mr", "mri", "scan", "radiology", "imaging", "abd", "pelvis", "chest"]):
        return "radiology_ct"
    if any(k in n for k in ["cyto", "cytology"]):
        return "cytology"
    if any(k in n for k in ["path", "surg", "biopsy", "histology", "specimen", "gross", "microscopic", "turbt"]):
        return "pathology"
    return "note"

def derive_pid_from_extracted(j: dict, fallback_filename: str) -> str:
    # Prefer MRN if it looks like an ID
    mrn = (j.get("mrn") or "").strip()
    if mrn and re.fullmatch(r"[A-Za-z0-9\-]{6,}", mrn):
        return mrn
    # Else fallback to normalized "LAST,FIRST [DOB]" if present
    name = (j.get("patient_name") or "").strip()
    dob  = (j.get("dob") or "").strip()
    if name and dob:
        return f"{name} [{dob}]"
    if name:
        return name
    # Else use the leading token from filename
    stem = fallback_filename.rsplit(".", 1)[0]
    tok = re.split(r'[\s\-_]+', stem)
    return tok[0] if tok else stem

def facts_map_to_df():
    df = pd.DataFrame([asdict(v) for v in st.session_state.facts_map.values()])

    # Expand status fields for easy filtering
    if "recurrence" in df.columns:
        df["recurrence_flag"] = df["recurrence"].apply(lambda x: bool(isinstance(x, dict) and x.get("has_recurrence")))
        df["recurrence_date"] = df["recurrence"].apply(lambda x: x.get("date") if isinstance(x, dict) else None)
        df["recurrence_site"] = df["recurrence"].apply(lambda x: x.get("site") if isinstance(x, dict) else None)
    if "metastasis" in df.columns:
        df["metastasis_flag"] = df["metastasis"].apply(lambda x: bool(isinstance(x, dict) and x.get("has_metastasis")))
        df["metastasis_sites"] = df["metastasis"].apply(lambda x: ", ".join(x.get("sites", [])) if isinstance(x, dict) else None)
    st.session_state.facts = df

def safe_json_loads(s: str) -> dict:
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r'\{.*\}', s, flags=re.S)
        return json.loads(m.group(0)) if m else {}

# ===================== LLM PROMPTS (brace-safe) =====================
EXTRACT_SYSTEM = (
    "You are a clinical information extraction service. "
    "Return ONE JSON object only — no extra text — matching the schema."
)

# Doubled braces to keep .format() happy
EXTRACT_USER_TMPL = """From this clinical {kind} report text, extract the following fields into JSON.
Unknown fields must be null or sensible defaults.

{{
  "patient_name": null,              // e.g., "Watson, Johanna K"
  "mrn": null,                       // medical record number if present (e.g., '88848822')
  "dob": null,                       // patient date of birth as written
  "exam_date": null,                 // the exam/report date, normalized if you can (YYYY-MM-DD) else as written
  "cancer_type": null,               // e.g., "Renal cell carcinoma", "Bladder cancer", etc., if a convincing primary is implied
  "tnm_stage": null,                 // e.g., "pT3a N0 M0" if present (pathology); PET/CT may omit
  "tumor_size_cm": null,             // numeric cm if a dominant primary lesion is measured
  "diagnosis_date": null,
  "surgery_date": null,

  "nodal_disease": null,             // true/false if nodes involved or metastatic; null if not stated
  "metastasis": {{                   // for PET/CT or CT when mets are described
    "has_metastasis": false,
    "sites": []                      // from: ["Liver","Lung","Lymph nodes","Bone","Brain","Adrenal","Other"]
  }},
  "recurrence": {{                   // ONLY true if text implies recurrence/progression/new/worsening compared with prior
    "has_recurrence": false,
    "date": null,                    // if a comparison date or study date implies recurrence timing
    "site": null,                    // best single site if stated
    "reason": null                   // e.g., "progression", "new lesions", "interval increase"; DO NOT set true for initial staging
  }}
}}

Rules:
- Prefer MRN and patient name exactly as printed if you can find them.
- PET/CT reports often list **initial staging** and **metastatic** disease; in such cases set metastasis.has_metastasis=true with appropriate sites, but keep recurrence.has_recurrence=false unless the text says progression, recurrence, or new/worsening compared to prior.
- In CT impressions, phrases like "lymphadenopathy", "metastatic", "new lesions", "interval increase/worsening" suggest nodal disease, metastasis, or recurrence (only mark recurrence if compared to prior or explicitly stated).
- Cytology/pathology reports that say "Negative for malignant cells" should NOT set cancer_type nor recurrence/metastasis true.

TEXT:
<<<
{body}
>>>
"""

PARSE_SYSTEM = "You convert cohort questions into compact JSON filters. Return ONE JSON object, no prose."

PARSE_USER_TMPL = """Convert this question into filters.

QUESTION:
{q}

Return JSON with this exact shape:
{{
  "filters": {{
    "cancer_type_contains": null,     // e.g., "bladder", "renal", "urothelial" (case-insensitive substring) or null
    "tnm_include": [],                // e.g., ["PT3","PT4"] (normalize T-specs T3/T4/pT3/pT4 -> PT3/PT4)
    "recurrence_required": false,     // true if "WITH recurrence/progression" is requested
    "metastasis_required": false,     // true if they ask for metastasis/metastatic disease
    "met_sites_any": [],              // e.g., ["Liver","Lung","Lymph nodes","Bone","Brain","Adrenal","Other"]
    "size_cm": null                   // {{ "op": ">="|"<"|"<="|">"|"==", "value": number }}
  }}
}}
Interpretation rules:
- If the question mentions "bladder", "urothelial", "renal", "kidney", set cancer_type_contains accordingly (one best term).
- If it says "WITH recurrence" or "progression", set recurrence_required=true.
- If it says "metastatic" or specific sites (liver/lung/nodes/bone/brain/adrenal), set metastasis_required=true and include those sites in met_sites_any.
- If a cm threshold is mentioned ("size >= 3 cm", or just "3 cm"), fill size_cm (default op \">=\" if only a number is given).
- Normalize TNM T-specs to uppercase "PT3","PT4" etc.
"""

# ===================== LLM CALLS =====================
def llm_extract_struct(client: OpenAI, model: str, body: str, kind: str) -> dict:
    body = (body or "")[:max_chars]  # truncate long docs
    messages = [
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user", "content": EXTRACT_USER_TMPL.format(kind=kind, body=body)}
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
    st.subheader("📥 Upload PDFs")
    uploads = st.file_uploader(
        "Filenames can be anything — the app will read MRN/Name from inside the PDF.",
        type=["pdf"], accept_multiple_files=True
    )
    col1, col2 = st.columns(2)
    with col1:
        ingest_btn = st.button("Process uploads", use_container_width=True)
    with col2:
        clear_btn = st.button("Clear all", use_container_width=True)

if clear_btn:
    st.session_state.facts_map.clear()
    st.session_state.facts = pd.DataFrame()
    st.success("Cleared in-memory data.")

# ===================== INGEST (LLM-EXTRACT) =====================
def safe_append(lst: Optional[List], item):
    if lst is None:
        return [item]
    lst.append(item)
    return lst

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

                kind = guess_kind_from_name(up.name)  # initial guess, not critical
                try:
                    j = llm_extract_struct(client, model, text, kind=kind)
                except Exception as e:
                    st.error(f"LLM extraction failed for {up.name}: {e}")
                    continue

                # Build/merge PatientFacts
                pid = derive_pid_from_extracted(j, up.name)
                pf = st.session_state.facts_map.get(pid, PatientFacts(patient_id=pid, treatments=[], sources=[]))

                # identity
                pf.mrn = j.get("mrn") or pf.mrn
                pf.name = j.get("patient_name") or pf.name
                pf.dob  = j.get("dob") or pf.dob

                # core fields
                pf.cancer_type = j.get("cancer_type") or pf.cancer_type
                pf.tnm_stage   = j.get("tnm_stage") or pf.tnm_stage

                val = j.get("tumor_size_cm")
                if isinstance(val, (int, float, str)) and str(val).strip():
                    try:
                        valf = float(val)
                        if pf.tumor_size_cm is None or valf > float(pf.tumor_size_cm):
                            pf.tumor_size_cm = valf
                    except Exception:
                        pass

                if j.get("diagnosis_date"): pf.diagnosis_date = j["diagnosis_date"]
                if j.get("surgery_date"):   pf.surgery_date   = j["surgery_date"]

                # nodal / mets / recurrence
                nd = j.get("nodal_disease")
                if isinstance(nd, bool):
                    pf.nodal_disease = nd if pf.nodal_disease is None else (pf.nodal_disease or nd)

                mets = j.get("metastasis") or {}
                if isinstance(mets, dict) and mets.get("has_metastasis"):
                    prev_sites = set((pf.metastasis or {}).get("sites", []))
                    new_sites  = set(mets.get("sites", []))
                    pf.metastasis = {
                        "has_metastasis": True,
                        "sites": sorted(prev_sites.union(new_sites))
                    }
                elif pf.metastasis is None:
                    pf.metastasis = {"has_metastasis": False, "sites": []}

                rec = j.get("recurrence") or {}
                if isinstance(rec, dict) and rec.get("has_recurrence") is True:
                    pf.recurrence = {
                        "has_recurrence": True,
                        "date": rec.get("date"),
                        "site": rec.get("site"),
                        "reason": rec.get("reason")
                    }
                    if not pf.last_ct_date and j.get("exam_date"):
                        pf.last_ct_date = j.get("exam_date")
                elif isinstance(rec, dict) and rec.get("has_recurrence") is False:
                    if pf.recurrence is None:
                        pf.recurrence = {"has_recurrence": False, "date": None, "site": None, "reason": None}

                # sources
                dtype = ("PET/CT" if "pet" in kind else
                         "CT/CTA" if "ct" in kind else
                         "Cytology" if "cyto" in kind else
                         "Pathology" if "path" in kind else
                         "Note")
                pf.sources = safe_append(pf.sources, {"type": dtype, "origin": "upload", "name": up.name})

                st.session_state.facts_map[pid] = pf
                added += 1

        facts_map_to_df()
        st.success(f"Processed {added} PDFs. Patients in memory: {len(st.session_state.facts)}")

# ===================== LLM PROMPTS =====================
PARSE_SYSTEM = "You convert cohort questions into compact JSON filters. Return ONE JSON object, no prose."

PARSE_USER_TMPL = """Convert this question into filters.

QUESTION:
{q}

Return JSON with this exact shape:
{{
  "filters": {{
    "cancer_type_contains": null,     // e.g., "bladder", "renal", "urothelial" (case-insensitive substring) or null
    "tnm_include": [],                // e.g., ["PT3","PT4"] (normalize T-specs T3/T4/pT3/pT4 -> PT3/PT4)
    "recurrence_required": false,     // true if "WITH recurrence/progression" is requested
    "metastasis_required": false,     // true if they ask for metastasis/metastatic disease
    "met_sites_any": [],              // e.g., ["Liver","Lung","Lymph nodes","Bone","Brain","Adrenal","Other"]
    "size_cm": null                   // {{ "op": ">="|"<"|"<="|">"|"==", "value": number }}
  }}
}}
Interpretation rules:
- If the question mentions "bladder", "urothelial", "renal", "kidney", set cancer_type_contains accordingly (one best term).
- If it says "WITH recurrence" or "progression", set recurrence_required=true.
- If it says "metastatic" or specific sites (liver/lung/nodes/bone/brain/adrenal), set metastasis_required=true and include those sites in met_sites_any.
- If a cm threshold is mentioned ("size >= 3 cm", or just "3 cm"), fill size_cm (default op \">=\" if only a number is given).
- Normalize TNM T-specs to uppercase "PT3","PT4" etc.
"""

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

# ===================== LLM QUERY PARSING + FILTERING =====================
def apply_filters(df: pd.DataFrame, fjson: dict) -> pd.DataFrame:
    out = df.copy()
    f = (fjson or {}).get("filters", {})

    # cancer type substring
    cstr = (f.get("cancer_type_contains") or "").strip().lower()
    if cstr and "cancer_type" in out.columns:
        out = out[out["cancer_type"].astype(str).str.lower().str.contains(cstr, na=False)]

    # TNM include (list like ["PT3","PT4"])
    tnm_list = f.get("tnm_include") or []
    if tnm_list and "tnm_stage" in out.columns:
        mask = pd.Series(False, index=out.index)
        for t in tnm_list:
            t_norm = str(t).upper()
            rx = r'\b' + re.escape(t_norm) + r'\b'
            mask = mask | out["tnm_stage"].astype(str).str.upper().str.replace(" ", "", regex=False).str.contains(rx, regex=True, na=False)
        out = out[mask]

    # recurrence
    if f.get("recurrence_required") and "recurrence" in out.columns:
        out = out[out["recurrence"].apply(lambda x: bool(isinstance(x, dict) and x.get("has_recurrence")))]

    # metastasis overall
    if f.get("metastasis_required") and "metastasis" in out.columns:
        out = out[out["metastasis"].apply(lambda x: bool(isinstance(x, dict) and x.get("has_metastasis")))]

    # metastasis sites any
    met_sites_any = f.get("met_sites_any") or []
    if met_sites_any and "metastasis" in out.columns:
        want = set([s.lower() for s in met_sites_any])
        def has_any_sites(m):
            if not isinstance(m, dict): return False
            sites = [s.lower() for s in m.get("sites", [])]
            return bool(want.intersection(sites))
        out = out[out["metastasis"].apply(has_any_sites)]

    # size threshold
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

    # columns: prioritize key clinical fields
    cols = ["patient_id","mrn","name","dob",
            "cancer_type","tnm_stage","tumor_size_cm",
            "diagnosis_date","surgery_date","last_ct_date",
            "recurrence_flag","recurrence_date","recurrence_site",
            "metastasis_flag","metastasis_sites",
            "nodal_disease","sources"]
    # ensure the expanded columns exist
    if "recurrence_flag" not in out.columns and "recurrence" in out.columns:
        out["recurrence_flag"] = out["recurrence"].apply(lambda x: bool(isinstance(x, dict) and x.get("has_recurrence")))
        out["recurrence_date"] = out["recurrence"].apply(lambda x: x.get("date") if isinstance(x, dict) else None)
        out["recurrence_site"] = out["recurrence"].apply(lambda x: x.get("site") if isinstance(x, dict) else None)
    if "metastasis_flag" not in out.columns and "metastasis" in out.columns:
        out["metastasis_flag"] = out["metastasis"].apply(lambda x: bool(isinstance(x, dict) and x.get("has_metastasis")))
        out["metastasis_sites"] = out["metastasis"].apply(lambda x: ", ".join(x.get("sites", [])) if isinstance(x, dict) else None)
    cols = [c for c in cols if c in out.columns]
    return out.loc[:, cols]

# ===================== MAIN QUERY UI =====================
st.markdown("### 🔎 Ask a question (LLM parsed)")
q_default = "Renal or bladder cancer with metastasis to liver or lung; include TNM and size >= 3 cm"
question = st.text_input("Natural-language query", value=q_default)

colA, colB = st.columns([1,1])
with colA:
    run_btn = st.button("Run query with LLM", type="primary")
with colB:
    export_all_btn = st.button("Download full dataset CSV")

if run_btn:
    if st.session_state.facts.empty:
        st.warning("No data yet. Upload PDFs in the sidebar.")
    else:
        client = get_openai_client()
        if client is None:
            st.stop()
        try:
            with st.spinner("Parsing question with LLM..."):
                parsed = llm_parse_query(client, model, question)
            st.caption(f"Parsed filters: `{json.dumps(parsed, ensure_ascii=False)}`")
            res = apply_filters(st.session_state.facts, parsed)
            st.success(f"{len(res)} rows")
            st.dataframe(res, use_container_width=True, height=420)
            st.download_button("Download results CSV", data=res.to_csv(index=False).encode("utf-8"),
                               file_name="cohort.csv", mime="text/csv")
        except Exception as e:
            st.error(f"LLM query parsing failed: {e}")

if export_all_btn:
    if st.session_state.facts.empty:
        st.warning("No data to export yet.")
    else:
        st.download_button("Download full dataset CSV",
                           data=st.session_state.facts.to_csv(index=False).encode("utf-8"),
                           file_name="patient_facts_full.csv", mime="text/csv")

st.markdown("""
#### Example queries
- **Bladder cancer WITH recurrence; show site**  
- **Renal cancer with metastasis to liver or lung; include TNM and size >= 3 cm**  
- **Any cancer with mediastinal lymph node disease**  
- **Bladder or renal with stage T3 or T4**  
""")
