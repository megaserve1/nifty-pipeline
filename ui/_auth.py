"""ui/_auth.py -- one shared password gate, called first thing on EVERY page.

WHY IT IS ON EVERY PAGE, NOT JUST HOME
    streamlit's multipage router serves pages/1_Build_Dataset.py at its own URL. anyone who knows
    (or guesses) that URL reaches the page directly without ever loading Home.py. a gate on Home
    alone protects nothing. so require_auth() is the first call in every page file.

WHERE THE PASSWORD COMES FROM
    the NIFTY_UI_PASSWORD environment variable. never a file in the repo -- a password committed
    to git is a password published. if it is not set the app REFUSES to serve anything rather
    than falling open, because the failure mode of "no password set" must not be "no password".

    set it in the same shell that launches streamlit:
        export NIFTY_UI_PASSWORD='something-long'
        nohup final_venv/bin/streamlit run ui/Home.py > /tmp/streamlit.log 2>&1 &

WHAT IT IS AND IS NOT
    it is one shared password over a link, which is enough to keep strangers and scanners out.
    it is NOT per-user accounts and it does NOT encrypt anything -- on a plain http:// address the
    password crosses the network readable. over the LAN that is fine. over a public tunnel, use
    the https:// address the tunnel gives you.
"""
import hmac
import os

import streamlit as st

ENV_VAR = "NIFTY_UI_PASSWORD"


def _secret() -> str:
    """the password, from the environment OR from streamlit's secrets store.

    LOCALLY it comes from the shell (export NIFTY_UI_PASSWORD=...). there is no shell on a hosted
    runner, so a deployment there reads st.secrets instead -- set it in the host's Secrets box as
        NIFTY_UI_PASSWORD = "..."
    st.secrets raises rather than returning None when no secrets file exists, which is the normal
    case on this machine, so it is guarded.
    """
    v = os.environ.get(ENV_VAR, "")
    if v:
        return v
    try:
        return str(st.secrets[ENV_VAR])
    except Exception:
        return ""


def require_auth():
    """stop the page unless the visitor has entered the password. returns nothing on success."""
    secret = _secret()

    if not secret:
        # fail CLOSED. an unset variable is a mistake, not permission to skip the gate.
        st.error(f"**{ENV_VAR} is not set**, so this app will not serve any page.", icon="🔒")
        st.caption("set it in the shell that starts streamlit, then restart:")
        st.code(f"export {ENV_VAR}='choose-a-long-one'\n"
                f"nohup final_venv/bin/streamlit run ui/Home.py > /tmp/streamlit.log 2>&1 &",
                language="bash")
        st.stop()

    if st.session_state.get("_authed"):
        return

    st.markdown("### 🔒  Nifty pipeline")
    st.caption("this page can start jobs that cost money, so it asks for the password first.")
    with st.form("login"):
        typed = st.text_input("Password", type="password")
        ok = st.form_submit_button("Enter", type="primary")
    if ok:
        # compare_digest, not == : a plain comparison returns as soon as two characters differ,
        # and that timing difference leaks the password one character at a time.
        if hmac.compare_digest(typed, secret):
            st.session_state["_authed"] = True
            st.rerun()
        else:
            st.error("wrong password")
    st.stop()


def logout_button():
    """optional, for the sidebar."""
    if st.session_state.get("_authed") and st.sidebar.button("Log out"):
        st.session_state["_authed"] = False
        st.rerun()
