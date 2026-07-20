import hmac
import os
import streamlit as st

def render_login() -> bool:
    expected_user = os.getenv("DEMO_USERNAME", "student")
    expected_password = os.getenv("DEMO_PASSWORD", "capstone123")
    if st.session_state.get("authenticated"):
        with st.sidebar:
            st.success(f"Signed in as {st.session_state.get('username', 'user')}")
            if st.button("Log out"):
                st.session_state.clear()
                st.rerun()
        return True

    st.title("Marketing RAG Assistant")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", use_container_width=True)
    if submitted:
        if hmac.compare_digest(username, expected_user) and hmac.compare_digest(password, expected_password):
            st.session_state.authenticated = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Incorrect username or password.")
    st.info("Demo login: student / capstone123")
    return False

