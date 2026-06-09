import streamlit as st

pg = st.navigation([
    st.Page("Description_and_Templates.py", title="Description & Templates", icon="🏠"),
    st.Page("pages/1_Expertise_Matching.py", title="Expertise Matching", icon="🔬"),
    st.Page("pages/2_Resubmission_Matcher.py", title="Resubmission Matcher", icon="🔄"),
    st.Page("pages/3_Best_Partner_Identification.py", title="Best Partner Identification", icon="🤝"),
])
pg.run()
