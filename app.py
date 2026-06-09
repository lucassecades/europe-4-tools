#!/usr/bin/env python3
"""
EU Research Tools — main entry point for the Streamlit multi-page app.
"""
import streamlit as st

st.set_page_config(page_title="EU Research Tools", page_icon="🇪🇺", layout="wide")

st.title("🇪🇺 EU Research Tools")
st.markdown("""
Select a tool from the sidebar:

| # | Tool | What it does |
|---|------|-------------|
| 1 | **Expertise Matching** | Match EU calls to your team's expertise areas |
| 2 | **Resubmission Matcher** | Score open calls against a project abstract for resubmission |
| 3 | **Best Partner Identification** | Find the best EU project partners for your abstract |
| 4 | **EU Portal Screening** | Extract calls from saved EU portal HTML pages |
""")
