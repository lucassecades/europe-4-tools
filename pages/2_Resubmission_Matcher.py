#!/usr/bin/env python3
"""
EU Open Calls Resubmission Matcher â€” Streamlit version
"""
import io
import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="Resubmission Matcher", page_icon="ðŸ”„", layout="wide")
DEFAULT_EXCEL = Path(__file__).parent.parent / "5. data" / "EU calls data" / "EUcalls-June26.xlsx"

st.title("ðŸ”„ EU Open Calls Resubmission Matcher")
st.markdown("Score EU open calls against your project abstract to find the best resubmission targets.")

# â”€â”€ Optional heavy deps â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

_MODEL = None

def get_model():
    global _MODEL
    if _MODEL is None:
        with st.spinner("Loading embedding model (first run only)â€¦"):
            _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def compute_similarity(text1: str, text2: str) -> float:
    if not text1 or not text2:
        return 0.0
    text1, text2 = str(text1).strip(), str(text2).strip()
    if EMBEDDING_AVAILABLE:
        try:
            m = get_model()
            e1, e2 = m.encode(text1), m.encode(text2)
            return float(max(0.0, min(1.0, sk_cosine([e1], [e2])[0][0])))
        except Exception:
            pass
    if RAPIDFUZZ_AVAILABLE:
        ts = fuzz.token_set_ratio(text1, text2) / 100.0
        pa = fuzz.partial_ratio(text1, text2) / 100.0
        si = fuzz.ratio(text1, text2) / 100.0
        return max(0.0, min(1.0, ts * 0.5 + pa * 0.3 + si * 0.2))
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def map_columns(columns):
    norm = {str(c).strip().lower(): c for c in columns}
    def pick(candidates):
        for n in candidates:
            if n in norm:
                return norm[n]
        return None
    return {
        "call_id":     pick(["call", "call id", "id call", "call_id", "identifier", "id"]),
        "title":       pick(["title", "call title", "project title"]),
        "description": pick(["description", "call description", "details", "summary", "abstract"]),
        "link":        pick(["url", "link", "call url", "call link", "web", "website"]),
        "deadline":    pick(["deadline", "deadline date", "next deadline", "closing date"]),
        "budget":      pick(["budget"]),
        "type":        pick(["type", "action type", "type of action"]),
    }


def read_text(file) -> str:
    for enc in ["utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "latin-1"]:
        try:
            file.seek(0)
            return file.read().decode(enc)
        except Exception:
            continue
    raise ValueError("Cannot decode abstract file")


def extract_keywords(text: str) -> list[str]:
    m = re.compile(r"^\s*keywords\s*:\s*(.+)$", re.I | re.M | re.DOTALL).search(text)
    if not m:
        return []
    block = re.split(r"\n\s*\n", m.group(1).strip(), maxsplit=1)[0]
    return [k.strip() for k in re.split(r"[;,]\s*", block) if k.strip()]

# â”€â”€ Inputs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

col1, col2 = st.columns(2)
with col1:
    st.subheader("Step 1 â€” Excel file")
    if DEFAULT_EXCEL.exists():
        st.success(f"Using server file: `{DEFAULT_EXCEL.name}`")
        excel_upload = None
    else:
        excel_upload = st.file_uploader("EU calls Excel (.xlsx)", type=["xlsx"])
with col2:
    st.subheader("Step 2 â€” Abstract")
    st.caption("Recommended: 150â€“400 words (description + keywords)")
    abstract_input_mode = st.radio("Input method", ["Upload .txt file", "Paste text"], horizontal=True, key="abs_mode_2")
    if abstract_input_mode == "Upload .txt file":
        abstract_upload = st.file_uploader("Abstract text file (.txt)", type=["txt"])
        abstract_pasted = None
    else:
        abstract_upload = None
        abstract_pasted = st.text_area("Paste your abstract here", height=200, placeholder="Description\n\nYour project description...\n\nKeywords\n\nkeyword1, keyword2, keyword3")

# â”€â”€ Run â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

st.subheader("Step 3 â€” Run")

excel_ready = DEFAULT_EXCEL.exists() or excel_upload
abstract_ready = abstract_upload or (abstract_pasted and abstract_pasted.strip())
if st.button("â–¶ Run Analysis", disabled=(not excel_ready or not abstract_ready)):
    if DEFAULT_EXCEL.exists():
        df = pd.read_excel(DEFAULT_EXCEL)
    else:
        df = pd.read_excel(excel_upload)
    st.info(f"Loaded {len(df)} rows from Excel")

    abstract_text = (read_text(abstract_upload) if abstract_upload else abstract_pasted).strip()
    if not abstract_text:
        st.error("Abstract is empty")
        st.stop()

    kws = extract_keywords(abstract_text)
    if kws:
        abstract_text += "\n\nKeywords: " + "; ".join(kws)

    col_map = map_columns(df.columns)
    if not col_map.get("description"):
        st.error("No description column found in the Excel file")
        st.stop()

    progress = st.progress(0)
    status = st.empty()
    rows = []
    total = len(df)

    for i, row in enumerate(df.itertuples(index=False)):
        title       = getattr(row, col_map["title"],       "") if col_map.get("title")       else ""
        call_id     = getattr(row, col_map["call_id"],     "") if col_map.get("call_id")     else ""
        description = getattr(row, col_map["description"], "") if col_map.get("description") else ""
        link        = getattr(row, col_map["link"],        "") if col_map.get("link")        else ""
        deadline    = getattr(row, col_map["deadline"],    "") if col_map.get("deadline")    else ""
        budget      = getattr(row, col_map["budget"],      "") if col_map.get("budget")      else ""
        action_type = getattr(row, col_map["type"],        "") if col_map.get("type")        else ""

        label = str(title) if str(title).strip() else str(call_id)
        status.text(f"Comparing {i + 1}/{total}: {label}")

        score = compute_similarity(abstract_text, f"{title}\n{description}") * 100.0
        rows.append({
            "ID call": call_id, "title": title, "description": description,
            "budget": budget, "type": action_type, "link": link,
            "deadline": deadline, "similarity score": score,
        })
        progress.progress((i + 1) / total)

    status.empty()

    result = pd.DataFrame(rows)
    result = result[result["similarity score"].fillna(0) > 0]
    result = result.sort_values("similarity score", ascending=False)
    result["similarity score"] = result["similarity score"].map(lambda v: f"{v:.2f}".replace(".", ","))

    buf = io.BytesIO()
    result.to_excel(buf, index=False)
    buf.seek(0)

    st.success(f"Done â€” {len(result)} calls scored.")
    st.dataframe(result[["ID call", "title", "similarity score", "deadline"]].head(20), use_container_width=True)
    st.download_button(
        "â¬‡ Download analysis Excel",
        data=buf,
        file_name=f"EU open calls - analysis - {Path(abstract_upload.name).stem if abstract_upload else 'abstract'}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
