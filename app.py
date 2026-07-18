import streamlit as st

from ui.chat import render_chat
from ui.history import init_history
from ui.sidebar import render_sidebar

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_history()

selected_source = render_sidebar()

st.title("DocuMind AI")
st.caption("Enterprise Document Intelligence Platform")
st.divider()

render_chat(source_filter=selected_source)
