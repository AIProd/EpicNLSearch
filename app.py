import io
import re
import json
import time
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from pypdf import PdfReader

# ============ UI CONFIG ============
st.set_page_config(page_title="EPIC NL Search - PoC", page_icon="🩺", layout="wide")
st.title("🩺 EPIC NL Search — PoC")

st.markdown("""
This PoC turns unstructured clinical **PDFs** into a **searchable patient table** you can query in natural language.

**What it extracts now**
- From **pathology**: TNM stage, tumor size (cm), diagnosis date, (optionally cancer type when text says *bladder/urothelial*).
- From **radiology**: recurrence/progression (with **negation** handling), recurrence site, and report date (if present).

**How to use**
1) Upload de-identified PDFs (pathology + radiology) in the sidebar.
2) Ask a question (examples below).
3) Download the CSV for your cohort.
""")

# ============ DATA MODELS & HELPERS ============
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
    recurrence: Optional[Dict] = None
    last_ct_date: Optional[str] = None
    death_date: Optional[str] = None
    sources: List[Dict] = None

# regex patterns
TNM_PAT = re.compile(r'\b(p?T[0-4][abc]?)[,\s;/]*(N[0-3X][abc]?)[,\s;/]*(M[01X])\b', re.I)
T_PAT   = re.compile(r'\b(p?T[0-4][abc]?)\b', re.I)
N_PAT   = re.compile(r'\bN[0-3X][abc]?\b', re.I)
M_PAT   = re.compile(r'\bM[01X]\b', re.I)

SIZE_PAT = re.compile(r'(tumou?r|mass|lesion)[^\.:\n]{0,40}?\b(?:size|measures?)?\b[^0-9]{0,10}?(\d+(?:\.\d+)?)\s*(cm|mm)\b', re.I)
DATE_PAT = re.compile(r'\b(20\d{2}|19\d{2})[-/\.](0?[1-9]|1[0-2])[-/\.](0?[1-9]|[12]\d|3[01])\b')

# robust negation for recurrence/progression
REC_NEG = re.compile(
    r'\b(no|without|absent|free of)\b.{0,20}\b('
    r'recurr|recurrence|progression|metastasis|metastatic|mets|lymphadenopathy'
    r')\b',
    re.I
)
REC_POS = re.compile(
    r'\b('
    r'recurr|recurrence|progression|metastasis|metastatic|mets|new\s+lesion|enlarging|lymphadenopathy'
    r')\b',
    re.I
)

SITE_MAP = [
    (re.compile(r'bladder|pelvis|pelvic', re.I), "Bladder/Pelvis (GU)"),
    (re.compile(r'lymph|node', re.I), "Lymph nodes"),
    (re.compile(r'lung|pulmonary|chest|pleura', re.I), "Distant (lung/chest)"),
    (re.compile(r'liver|hepatic|bone|brain|adrenal', re.I), "Distant (other)"),
]

def extract_tnm(text:str) -> Optional[str]:
    if not text: return None
    m = TNM_PAT.search(text)
    if m: return "".join([p.upper() for p in m.groups() if p])
    t = T_PAT.search(text)
    n = N_PAT.search(text)
    m = M_PAT.search(text)
    parts = [x.group(0).upper() for x in [t,n,m] if x]
    return "".join(parts) if parts else None

def extract_size_cm(text:str) -> Optional[float]:
    if not text: return None
    m = SIZE_PAT.search(text)
    if not m: return None
    _, val, unit = m.groups()
    size = float(val)
    return round(size if unit.lower()=="cm" else size/10.0, 2)

def extract_any_date(text:str) -> Optional[str]:
    m = DATE_PAT.search(text or "")
    return m.group(0) if m else None

def detect_recurrence(text:str) -> Tuple[bool, Optional[str], Optional[str]]:
    if not text: return (False,None,None)
    t = " ".join(text.split())
    if REC_NEG.search(t):  # any explicit negation overrides
        return (False, None, None)
    if not REC_POS.search(t):
        return (False, None, None)
    site = None
    for rx, label in SITE_MAP:
        if rx.search(t):
            site = label; break
    dt = extract_any_date(t)
    return (True, dt, site)

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
    # default to pathology
    return "pathology"

def derive_pid_from_name(name: str) -> str:
    stem = name.rsplit(".", 1)[0]
    tok = re.split(r'[\s\-_]+', stem)
    return tok[0] if tok else stem

def ensure_df():
    if "facts_map" not in st.session_state:
        st.session_state.facts_map = {}
    if "facts" not in st.session_state:
        st.session_state.facts = pd.DataFrame()

def facts_map_to_df():
    df = pd.DataFrame([asdict(v) for v in st.session_state.facts_map.values()])
    if "recurrence" in df.columns:
        df["recurrence_flag"] = df["recurrence"].apply(lambda x: bool(isinstance(x, dict) and x.get("has_recurrence")))
        df["recurrence_date"] = df["recurrence"].apply(lambda x: x.get("date") if isinstance(x, dict) else None)
        df["recurrence_site"] = df["recurrence"].apply(lambda x: x.get("site") if isinstance(x, dict) else None)
    st.session_state.facts = df

# ============ SIDEBAR: UPLOADS & DEMO DATA ============
with st.sidebar:
    st.header("📥 Ingest data")
    uploaded = st.file_uploader(
        "Upload de-identified PDFs (pathology & radiology). Filenames should include a patient ID.",
        type=["pdf"], accept_multiple_files=True
    )
    colA, colB = st.columns(2)
    with colA:
        ingest_btn = st.button("Process uploads", use_container_width=True)
    with colB:
        clear_btn = st.button("Clear all", use_container_width=True)

    st.markdown("---")
    st.subheader("⚡ Quick demo patients (no upload)")
    demo_btn = st.button("Add demo patients (BLD123 & BLD999)")

ensure_df()

if clear_btn:
    st.session_state.facts_map = {}
    st.session_state.facts = pd.DataFrame()
    st.success("Cleared all in-memory data.")

if ingest_btn and uploaded:
    added = 0
    for up in uploaded:
        b = up.read()
        text = pdf_to_text(b)
        if not text.strip():
            st.warning(f"Skipped (no extractable text): {up.name}")
            continue
        pid = derive_pid_from_name(up.name)
        kind = guess_kind_from_name(up.name)

        pf = st.session_state.facts_map.get(pid, PatientFacts(patient_id=pid, treatments=[], sources=[]))

        if kind == "pathology":
            tnm = extract_tnm(text)
            size = extract_size_cm(text)
            diag_date = extract_any_date(text)
            if re.search(r'\b(bladder|urothelial)\b', text, re.I):
                pf.cancer_type = pf.cancer_type or "Bladder cancer"
            if tnm and not pf.tnm_stage: pf.tnm_stage = tnm
            if size is not None and (pf.tumor_size_cm is None or size > pf.tumor_size_cm): pf.tumor_size_cm = size
            if diag_date and not pf.diagnosis_date: pf.diagnosis_date = diag_date
        else:
            has_rec, rec_dt, site = detect_recurrence(text)
            if has_rec:
                pf.recurrence = {"has_recurrence": True, "date": rec_dt, "site": site, "evidence": "radiology_positive_signal"}
                if not pf.last_ct_date and rec_dt: pf.last_ct_date = rec_dt

        if pf.sources is None: pf.sources = []
        pf.sources.append({"type": f"{kind}_report", "origin": "upload", "link": None})
        st.session_state.facts_map[pid] = pf
        added += 1

    facts_map_to_df()
    st.success(f"Processed {added} PDFs. Current patients: {len(st.session_state.facts)}")

if demo_btn:
    # Synthetic demo rows — mimic a bladder + recurrence and bladder without recurrence
    for pid, patho_txt, radio_txt in [
        ("BLD123",
         "Final Diagnosis: Invasive urothelial carcinoma (bladder). Pathologic stage: pT3a N0 M0. Tumor size 5.2 cm. Date: 2024-06-14.",
         "CT CHEST 2024-10-21. New lesion and mediastinal lymphadenopathy. Impression: progression consistent with recurrence."
        ),
        ("BLD999",
         "Urothelial carcinoma of the bladder. Pathologic stage: pT4 N1 M0. Tumor size 3.2 cm. Date: 2024-05-03.",
         "CT 2025-01-05. No lymphadenopathy. No evidence of recurrence or metastasis."
        )
    ]:
        pf = st.session_state.facts_map.get(pid, PatientFacts(patient_id=pid, treatments=[], sources=[]))
        # pathology
        tnm = extract_tnm(patho_txt); size = extract_size_cm(patho_txt); diag = extract_any_date(patho_txt)
        pf.cancer_type = pf.cancer_type or "Bladder cancer"
        if tnm and not pf.tnm_stage: pf.tnm_stage = tnm
        if size is not None and (pf.tumor_size_cm is None or size > pf.tumor_size_cm): pf.tumor_size_cm = size
        if diag and not pf.diagnosis_date: pf.diagnosis_date = diag
        if pf.sources is None: pf.sources = []
        pf.sources.append({"type":"pathology_report","origin":"demo","link":None})
        # radiology
        has_rec, rec_dt, site = detect_recurrence(radio_txt)
        if has_rec:
            pf.recurrence = {"has_recurrence": True, "date": rec_dt, "site": site, "evidence":"radiology_positive_signal"}
            if not pf.last_ct_date and rec_dt: pf.last_ct_date = rec_dt
        pf.sources.append({"type":"radiology_report","origin":"demo","link":None})
        st.session_state.facts_map[pid] = pf
    facts_map_to_df()
    st.success("Added demo patients BLD123 (recurrence) and BLD999 (no recurrence).")

# ============ QUERY ENGINE ============
WITH_RECURRENCE_PAT = re.compile(r'\b(with|has|having|show)\s+(recurr|progression)', re.I)
POS_RECURRENCE_PAT  = re.compile(r'\brecurrence[-\s]?positive\b', re.I)

def query_to_filters(q: str) -> Dict:
    ql = q.lower()
    f: Dict = {}
    if "bladder" in ql: f["cancer_type"] = "bladder"
    # accept both PT3 and T3, PT4 and T4
    if "pt3" in ql or re.search(r'\bt3\b', ql): f.setdefault("tnm_rx", []).append(r'\bP?T3\b')
    if "pt4" in ql or re.search(r'\bt4\b', ql): f.setdefault("tnm_rx", []).append(r'\bP?T4\b')
    # treat as FILTER only if phrased explicitly
    if WITH_RECURRENCE_PAT.search(ql) or POS_RECURRENCE_PAT.search(ql): f["recurrence"] = True
    # numeric size (>= N cm) or bare "N cm"
    m = re.search(r'([><]=?)\s*(\d+(?:\.\d+)?)\s*cm', ql)
    if m:
        op, val = m.groups(); f["size_op"] = op; f["size_thr"] = float(val)
    else:
        m2 = re.search(r'\b(\d+(?:\.\d+)?)\s*cm\b', ql)
        if m2: f["size_op"] = ">="; f["size_thr"] = float(m2.group(1))
    return f

def apply_filters(df: pd.DataFrame, f: Dict) -> pd.DataFrame:
    out = df.copy()
    if f.get("cancer_type") and "cancer_type" in out.columns:
        out = out[out["cancer_type"].astype(str).str.contains("bladder", case=False, na=False)]
    if "tnm_rx" in f and "tnm_stage" in out.columns:
        mask = pd.Series(False, index=out.index)
        for rx in f["tnm_rx"]:
            mask = mask | out["tnm_stage"].astype(str).str.contains(rx, na=False, regex=True, case=False)
        out = out[mask]
    if f.get("recurrence") and "recurrence" in out.columns:
        out = out[out["recurrence"].apply(lambda x: bool(isinstance(x, dict) and x.get("has_recurrence")))]
    if "size_thr" in f and "tumor_size_cm" in out.columns:
        size = pd.to_numeric(out["tumor_size_cm"], errors="coerce")
        thr = f["size_thr"]; op = f.get("size_op", ">=")
        if op == ">": out = out[size > thr]
        elif op == ">=": out = out[size >= thr]
        elif op == "<": out = out[size < thr]
        elif op == "<=": out = out[size <= thr]
        elif op == "==": out = out[size == thr]
    # nice column order
    cols = ["patient_id","age","bmi","tnm_stage","tumor_size_cm",
            "diagnosis_date","surgery_date","last_ct_date","death_date",
            "recurrence_flag","recurrence_date","recurrence_site","cancer_type","sources"]
    for c in ["recurrence_flag","recurrence_date","recurrence_site"]:
        if c not in out.columns and "recurrence" in out.columns:
            out["recurrence_flag"] = out["recurrence"].apply(lambda x: bool(isinstance(x, dict) and x.get("has_recurrence")))
            out["recurrence_date"] = out["recurrence"].apply(lambda x: x.get("date") if isinstance(x, dict) else None)
            out["recurrence_site"] = out["recurrence"].apply(lambda x: x.get("site") if isinstance(x, dict) else None)
            break
    cols = [c for c in cols if c in out.columns]
    return out.loc[:, cols]

# ============ MAIN QUERY UI ============
st.markdown("### 🔎 Ask a question")
default_q = "All bladder cancer patients with stage T3 or T4; include age, TNM, tumor size"
question = st.text_input("Natural-language query", value=default_q, placeholder="e.g., Patients WITH recurrence or progression; show site and last CT date")

col1, col2, col3 = st.columns([1,1,2])
with col1:
    run_btn = st.button("Run query", type="primary")
with col2:
    export_all_btn = st.button("Export full dataset (CSV)")

if run_btn:
    if st.session_state.facts.empty:
        st.warning("No data yet. Upload PDFs or click 'Add demo patients' in the sidebar.")
    else:
        f = query_to_filters(question)
        res = apply_filters(st.session_state.facts, f)
        st.success(f"{len(res)} rows")
        st.dataframe(res, use_container_width=True, height=420)
        csv_bytes = res.to_csv(index=False).encode("utf-8")
        st.download_button("Download results CSV", data=csv_bytes, file_name="cohort.csv", mime="text/csv")

if export_all_btn:
    if st.session_state.facts.empty:
        st.warning("No data to export yet.")
    else:
        csv_bytes = st.session_state.facts.to_csv(index=False).encode("utf-8")
        st.download_button("Download full dataset CSV", data=csv_bytes, file_name="patient_facts_full.csv", mime="text/csv")

# ============ EXAMPLES ============
st.markdown("""
#### Example queries
- **All bladder cancer patients with stage T3 or T4; include age, TNM, tumor size**
- **Bladder cancer patients WITH recurrence or progression; show site and last CT date**
- **Bladder cancer with stage T3 and size >= 3 cm**
- **Patients WITH recurrence or progression; show site and last CT date** (no bladder filter)
""")
