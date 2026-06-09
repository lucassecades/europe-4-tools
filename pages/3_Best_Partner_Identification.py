#!/usr/bin/env python3
"""
Best Partner / Project Identification — Streamlit version
"""
import collections
import io
import os
import re
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from auth import require_password

st.set_page_config(page_title="Best Partner Identification", page_icon="🤝", layout="wide")
require_password()
st.title("🤝 Best EU Partner / Project Identification")
st.markdown("Match your abstract against the EU projects database to find the best collaboration partners.")

ROOT_DIR = Path(__file__).parent.parent


def resolve_data_dir(root: Path) -> Path:
    for name in ("5. data", "4. data", "data"):
        p = root / name
        if p.is_dir():
            return p
    for p in root.iterdir():
        if p.is_dir() and p.name.lower().endswith("data"):
            return p
    return root / "5. data"


DATA_DIR = resolve_data_dir(ROOT_DIR)
XML_DIR = DATA_DIR / "Best partner data" / "XML eu projects"

# ── Inputs ────────────────────────────────────────────────────────────────────

SERVER_CSV = DATA_DIR / "Best partner data" / "EU projects csv.csv"
SERVER_XML = DATA_DIR / "Best partner data" / "XML eu projects"

st.subheader("Step 1 — EU Projects CSV")
if SERVER_CSV.exists():
    st.success("Using server file: `EU projects csv.csv`")
    csv_upload = None
else:
    csv_upload = st.file_uploader("EU projects CSV file (`EU projects csv.csv`)", type=["csv"])

st.subheader("Step 2 — XML files (for partner analysis)")
if SERVER_XML.exists():
    st.success("Using server XML files for partner analysis")
    xml_zip_upload = None
else:
    xml_zip_upload = st.file_uploader(
        "ZIP of the `XML eu projects` folder — needed for partner organisations analysis (optional)",
        type=["zip"]
    )
    st.caption("To create the ZIP: right-click the `XML eu projects` folder → Send to → Compressed (zipped) folder")

st.subheader("Step 3 — Abstract")
abstract_upload = st.file_uploader("Abstract text file (.txt)", type=["txt"])

st.subheader("Step 4 — Settings")
num_top = st.number_input("Number of top projects to analyse", min_value=1, max_value=500, value=50)

# ── Run ───────────────────────────────────────────────────────────────────────

st.subheader("Step 5 — Run")

if st.button("▶ Run Matching", disabled=(abstract_upload is None or csv_upload is None)):
    abstract_text = abstract_upload.read().decode("utf-8", errors="replace").strip()

    desc_m = re.search(r"Description:\s*(.*?)\nKeywords:", abstract_text, re.DOTALL | re.I)
    kw_m   = re.search(r"Keywords:\s*(.*)",               abstract_text, re.DOTALL | re.I)
    description = desc_m.group(1).strip() if desc_m else abstract_text
    keywords    = kw_m.group(1).strip()   if kw_m  else ""
    abstract_full = description + " " + keywords

    # Load CSV
    with st.spinner("Loading EU projects database…"):
        try:
            csv_source = SERVER_CSV if SERVER_CSV.exists() else csv_upload
            df = pd.read_csv(csv_source, sep=";", quotechar='"', header=0,
                             on_bad_lines="skip", dtype=str)
            valid_col_count = 20
            df = df[df.apply(lambda r: len(r) == valid_col_count, axis=1)]
        except Exception as e:
            st.error(f"Failed to load CSV: {e}")
            st.stop()

    st.info(f"Database loaded: {len(df)} projects")

    # Embeddings
    with st.spinner("Loading embedding model (first run only)…"):
        model = SentenceTransformer("all-MiniLM-L6-v2")

    abstract_emb = model.encode(abstract_full, convert_to_tensor=True)
    abs_kw_list  = [k.strip() for k in keywords.split(",") if k.strip()]
    abs_kw_text  = " ".join(abs_kw_list)
    abs_kw_emb   = model.encode(abs_kw_text, convert_to_tensor=True) if abs_kw_text else None

    progress = st.progress(0)
    status   = st.empty()
    scores, obj_scores, kw_scores = [], [], []
    total = len(df)

    for step, (_, row) in enumerate(df.iterrows()):
        status.text(f"Scoring project {step + 1}/{total}…")
        obj = str(row.get("objective", ""))
        proj_kw_raw = str(row.get("keywords", ""))

        obj_emb = model.encode(obj, convert_to_tensor=True)
        obj_sim = float(cosine_similarity(
            [abstract_emb.cpu().numpy()], [obj_emb.cpu().numpy()]
        )[0][0])

        proj_kw_list = [k.strip() for k in proj_kw_raw.split(",") if k.strip()]
        proj_kw_text = " ".join(proj_kw_list)
        if abs_kw_emb is not None and proj_kw_text:
            proj_kw_emb = model.encode(proj_kw_text, convert_to_tensor=True)
            kw_sim = float(cosine_similarity(
                [abs_kw_emb.cpu().numpy()], [proj_kw_emb.cpu().numpy()]
            )[0][0])
            n = len(proj_kw_list)
            avg_len  = sum(len(k) for k in proj_kw_list) / n if n else 0
            diversity = len(set(proj_kw_list)) / (n or 1)
            kw_conf  = min(1.0, 0.5 * (n / 10) + 0.3 * (avg_len / 10) + 0.2 * diversity)
        else:
            kw_sim, kw_conf = 0.0, 0.0

        w_obj = 0.7
        w_kw  = 0.3 * kw_conf
        total_w = w_obj + w_kw
        final = (w_obj * obj_sim + w_kw * kw_sim) / total_w
        scores.append(int(final * 100))
        obj_scores.append(round(obj_sim * 100, 2))
        kw_scores.append(round(kw_sim * 100, 2))
        progress.progress((step + 1) / total)

    status.empty()
    df["match_score"] = scores
    df["Objective similarity score"] = obj_scores
    df["Keywords score"] = kw_scores

    top = df.sort_values("match_score", ascending=False).head(int(num_top)).copy()

    # ── Excel 1: project matches ──────────────────────────────────────────────
    top_out = top[["rcn", "id", "title", "objective", "keywords",
                   "Keywords score", "Objective similarity score", "match_score"]].copy()
    top_out["match_score_normalized"] = top_out["match_score"] / 100.0
    top_out["Project Website"] = top_out["id"].apply(
        lambda x: f"https://cordis.europa.eu/project/id/{x}" if pd.notna(x) else ""
    )
    col_order = ["rcn", "id", "title", "Project Website", "objective", "keywords",
                 "Keywords score", "Objective similarity score", "match_score", "match_score_normalized"]
    top_out = top_out[col_order]

    buf1 = io.BytesIO()
    top_out.to_excel(buf1, index=False)
    buf1.seek(0)

    # ── Excel 2: organisation analysis ───────────────────────────────────────
    status2 = st.empty()
    orgs = []
    rcn_to_title = {str(r.rcn): str(r.title) for r in top.itertuples(index=False)}

    def parse_xmls_from_dir(xml_dir: Path):
        for step, row in enumerate(top.itertuples(index=False)):
            rcn = str(getattr(row, "rcn"))
            xml_path = xml_dir / f"project-rcn-{rcn}_en.xml"
            if not xml_path.exists():
                continue
            try:
                tree = ET.parse(xml_path)
                xroot = tree.getroot()
                for org in xroot.findall(".//organization"):
                    name_raw = (org.findtext("legalName") or "").strip()
                    name = name_raw.lower()
                    cat = org.find('.//relations/categories/category[@classification="organizationActivityType"]/title')
                    type_ = (cat.text or "").strip() if cat is not None else ""
                    url_elem = org.find(".//address/url")
                    website = (url_elem.text or "").strip() if url_elem is not None else ""
                    orgs.append({
                        "name": name, "display_name": name_raw, "type": type_,
                        "rcn": rcn, "score": float(getattr(row, "match_score")) / 100.0,
                        "website": website,
                    })
            except Exception:
                pass
            status2.text(f"Parsing XML {step + 1}/{len(top)}…")

    if SERVER_XML.exists():
        parse_xmls_from_dir(SERVER_XML)
    elif xml_zip_upload is not None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with zipfile.ZipFile(xml_zip_upload, "r") as zf:
                zf.extractall(tmp_dir)
            # find the actual XML folder inside the zip
            tmp_path = Path(tmp_dir)
            xml_dirs = [p for p in tmp_path.rglob("*.xml")]
            if xml_dirs:
                parse_xmls_from_dir(xml_dirs[0].parent)

    status2.empty()

    org_agg = collections.defaultdict(
        lambda: {"type": "", "projects": set(), "score": 0.0,
                 "display_name": "", "project_pairs": set(), "project_names": set(), "website": ""}
    )
    for o in orgs:
        k = o["name"]
        pair = (o["name"], o["rcn"])
        org_agg[k]["type"] = o["type"]
        org_agg[k]["projects"].add(o["rcn"])
        org_agg[k]["project_names"].add(f"{rcn_to_title.get(o['rcn'], '')} (RCN {o['rcn']})")
        if not org_agg[k]["display_name"]:
            org_agg[k]["display_name"] = o["display_name"]
        if not org_agg[k]["website"] and o.get("website"):
            org_agg[k]["website"] = o["website"]
        if pair not in org_agg[k]["project_pairs"]:
            org_agg[k]["score"] += o["score"]
            org_agg[k]["project_pairs"].add(pair)

    org_rows = [
        {
            "name": v["display_name"], "type": v["type"], "website": v["website"],
            "# of projects": len(v["projects"]), "score of pertinence": v["score"],
            "Involved projects": "; ".join(sorted(v["project_names"])),
        }
        for v in org_agg.values()
    ]
    orgs_df = pd.DataFrame(org_rows) if org_rows else pd.DataFrame(
        columns=["name", "type", "website", "# of projects", "score of pertinence", "Involved projects"]
    )
    if not orgs_df.empty:
        orgs_df = orgs_df.sort_values(["score of pertinence", "# of projects"], ascending=[False, False])
    buf2 = io.BytesIO()
    orgs_df.to_excel(buf2, index=False)
    buf2.seek(0)

    st.success(f"Done — top {len(top)} projects analysed.")
    if orgs_df.empty:
        st.warning("No partner organisations found — XML files are not available on the server. Only the project matches Excel will be complete.")

    abstract_base = re.sub(r"[^A-Za-z0-9 _-]", "", Path(abstract_upload.name).stem).strip() or "Abstract"

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇ Download project matches Excel",
            data=buf1,
            file_name=f"{abstract_base}-EU projects matchmaking.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with c2:
        st.download_button(
            "⬇ Download partners analysis Excel",
            data=buf2,
            file_name=f"{abstract_base}-EU partners analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    st.subheader("Top 10 project matches")
    st.dataframe(top_out[["title", "match_score", "Project Website"]].head(10), use_container_width=True)
    st.subheader("Top 10 partner organisations")
    st.dataframe(orgs_df[["name", "type", "# of projects", "score of pertinence", "website"]].head(10),
                 use_container_width=True)
