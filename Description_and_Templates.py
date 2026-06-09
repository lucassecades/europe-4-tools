import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

st.title("EU Research Tools")

st.markdown("""
| # | Tool | What it does |
|---|------|-------------|
| 1 | **Expertise Matching** | Upload expertise `.txt` files - matches EU calls to your team's expertise areas |
| 2 | **Resubmission Matcher** | Upload an abstract `.txt` file - scores open calls for resubmission |
| 3 | **Best Partner Identification** | Upload an abstract `.txt` file - finds the best EU project partners |
""")

st.markdown("---")
st.subheader("Example input files")
st.markdown("Download these to understand the expected file format before using the tools.")

EXAMPLES_DIR = Path(__file__).parent / "examples"
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Abstract file** - used in Tools 2 and 3")
    abstract_ex = EXAMPLES_DIR / "AUTOCRAFT_abstract_example.txt"
    if abstract_ex.exists():
        st.download_button(
            "Download abstract example (AUTOCRAFT)",
            data=abstract_ex.read_bytes(),
            file_name="AUTOCRAFT_abstract_example.txt",
            mime="text/plain",
        )
with col2:
    st.markdown("**Expertise file** - used in Tool 1 (one file per person)")
    expertise_ex = EXAMPLES_DIR / "Adrian_Munteanu_Expertise_example.txt"
    if expertise_ex.exists():
        st.download_button(
            "Download expertise example (Adrian Munteanu)",
            data=expertise_ex.read_bytes(),
            file_name="Adrian_Munteanu_Expertise_example.txt",
            mime="text/plain",
        )
