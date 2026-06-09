#!/usr/bin/env python3
"""
EU Calls Expertise Matcher â€” Streamlit version
"""
import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

sys.path.insert(0, str(ROOT_DIR / "1. Expertise matchmaking EU"))

DEFAULT_EXCEL = ROOT_DIR / "5. data" / "EU calls data" / "EUcalls-June26.xlsx"

st.set_page_config(page_title="Expertise Matching", page_icon="ðŸ”¬", layout="wide")
st.title("ðŸ”¬ EU Calls Expertise Matcher")
st.markdown("Upload expertise `.txt` files and run the analysis.")

# â”€â”€ Inputs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

col1, col2 = st.columns(2)

with col1:
    st.subheader("Step 1 â€” Excel file")
    if DEFAULT_EXCEL.exists():
        st.success(f"Using server file: `{DEFAULT_EXCEL.name}`")
        excel_upload = None  # will use default
    else:
        excel_upload = st.file_uploader("EU calls Excel (.xlsx)", type=["xlsx"])

with col2:
    st.subheader("Step 2 â€” Expertise files")
    expertise_uploads = st.file_uploader(
        "Expertise .txt files (one per expertise area)",
        type=["txt"],
        accept_multiple_files=True,
    )

# â”€â”€ Run â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

st.subheader("Step 3 â€” Run")

excel_ready = DEFAULT_EXCEL.exists() or excel_upload
if st.button("â–¶ Run Expertise Analysis", disabled=(not excel_ready or not expertise_uploads)):
    from qualitative_expertise_analysis import (
        analyze_expertise_files,
        validate_alignment_comprehensive,
        generate_qualitative_explanation,
        is_valid_description,
    )

    log = st.empty()
    lines: list[str] = []

    def log_line(msg: str):
        lines.append(msg)
        log.code("\n".join(lines), language=None)

    log_line("=" * 70)
    log_line("EU CALLS EXPERTISE MATCHER")
    log_line("=" * 70)

    # Load Excel
    if DEFAULT_EXCEL.exists():
        df = pd.read_excel(DEFAULT_EXCEL)
    else:
        df = pd.read_excel(excel_upload)
    log_line(f"Excel loaded: {len(df)} rows")

    # Write expertise files to a temp dir so the analysis function can read them
    import tempfile, os
    tmp = Path(tempfile.mkdtemp())
    for f in expertise_uploads:
        (tmp / f.name).write_bytes(f.read())
    log_line(f"Expertise files: {len(expertise_uploads)}")

    # Analyse
    expertise = analyze_expertise_files(tmp)
    log_line(f"Expertise areas found: {len(expertise)}")

    marked_indices = df[df.get("Interesting brubotics", pd.Series()).fillna("") == "Yes"].index.tolist()
    log_line(f"Marked calls: {len(marked_indices)}")

    if "Touchpoints" not in df.columns:
        df["Touchpoints"] = ""

    matches: list[dict] = []
    matched_expertises: set[str] = set()

    progress = st.progress(0)
    for step, idx in enumerate(marked_indices):
        call = df.loc[idx]
        call_id = call.get("Call", f"Row {idx}")
        title = str(call.get("Title", ""))
        description = str(call.get("Description", ""))

        if not is_valid_description(description):
            log_line(f"  Skipped {call_id} â€” no valid description")
        else:
            call_matches = []
            for exp_name, exp_data in expertise.items():
                validation = validate_alignment_comprehensive(title, description, exp_data)
                if validation["overall_pass"]:
                    confidence = validation["confidence"]
                    if exp_name not in matched_expertises:
                        matched_expertises.add(exp_name)
                        if f"{exp_name} - Project Alignment" not in df.columns:
                            df[f"{exp_name} - Project Alignment"] = ""
                        if f"{exp_name} - Confidence" not in df.columns:
                            df[f"{exp_name} - Confidence"] = ""
                    explanation = generate_qualitative_explanation(exp_data, validation, title, description)
                    df.at[idx, f"{exp_name} - Project Alignment"] = explanation
                    df.at[idx, f"{exp_name} - Confidence"] = f"{confidence:.1%}"
                    call_matches.append(exp_name)
                    matches.append({"call": call_id, "expertise": exp_name, "confidence": confidence})
                    log_line(f"  Match: {str(call_id)[:40]} â†’ {exp_name} ({confidence:.1%})")
            if call_matches:
                df.at[idx, "Touchpoints"] = "; ".join(call_matches)

        progress.progress((step + 1) / max(len(marked_indices), 1))

    log_line(f"\nTotal matches: {len(matches)}")
    log_line("=" * 70)

    # Filter: only rows where at least one expertise confidence column is filled
    confidence_cols = [c for c in df.columns if c.endswith("- Confidence")]
    if confidence_cols:
        mask = df[confidence_cols].apply(lambda row: row.astype(str).str.strip().ne("").any(), axis=1)
        df = df[mask]

    # Output
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)
    st.success(f"Analysis complete â€” {len(matches)} matches found across {len(df)} calls.")
    expert_names = "_".join(Path(f.name).stem for f in expertise_uploads)
    st.download_button(
        "â¬‡ Download result Excel",
        data=buf,
        file_name=f"EUcalls-June26_expertise_analysis_{expert_names}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
