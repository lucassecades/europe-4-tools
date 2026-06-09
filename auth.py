import streamlit as st

PASSWORD = "brubotics-2026"

def require_password():
    if st.session_state.get("authenticated"):
        return
    st.title("🔒 EU Research Tools")
    pwd = st.text_input("Password", type="password")
    if st.button("Enter"):
        if pwd == PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Wrong password")
    st.stop()
