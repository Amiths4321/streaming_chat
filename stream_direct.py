# stream_direct.py
# streamlit run stream_direct.py
# This version streams directly without FastAPI — great for learning

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from stream_rag import stream_rag_answer

st.set_page_config(page_title="Direct Streaming", page_icon="⚡")
st.title("⚡ Direct Streaming — no FastAPI")
st.caption("Streams directly from Ollama via Python generator")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # st.write_stream() handles generators natively
        # It renders each yielded string as it arrives
        response = st.write_stream(
            stream_rag_answer(prompt, st.session_state.messages[-6:])
        )

    st.session_state.messages.append({
        "role":    "assistant",
        "content": response
    })