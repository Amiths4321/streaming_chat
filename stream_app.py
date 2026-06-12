# stream_app.py
# streamlit run stream_app.py

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests
import json
import streamlit as st

API_URL = "http://localhost:8001"

st.set_page_config(
    page_title="Streaming AI Chat",
    page_icon="⚡",
    layout="wide"
)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state: st.session_state.messages = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ Streaming AI Chat")
    st.caption("Real-time token streaming from Qwen2.5-VL")

    use_rag = st.toggle("Use knowledge base", value=True)

    st.divider()
    st.markdown("**How streaming works:**")
    st.caption("1. Your question → FastAPI")
    st.caption("2. FastAPI → Ollama (stream=True)")
    st.caption("3. Each token → SSE event")
    st.caption("4. Streamlit renders live")

    st.divider()
    st.markdown("**Try these:**")
    examples = [
        "How many leave days do employees get?",
        "What tools does TechCorp use?",
        "Explain the performance review process",
        "What is the price of CloudSync Pro?",
        "Write a short poem about AI",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=ex):
            st.session_state["pending"] = ex

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("⚡ Real-time Streaming AI Chat")
st.caption("Tokens stream live as Qwen generates them — no more waiting")

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources", expanded=False):
                for s in msg["sources"]:
                    st.caption(f"[{s['source']}] {s['similarity']} — {s['text'][:100]}...")

# ── Input ─────────────────────────────────────────────────────────────────────
pending    = st.session_state.pop("pending", None)
user_input = st.chat_input("Ask anything...") or pending

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Stream assistant response
    with st.chat_message("assistant"):
        full_response = ""
        placeholder   = st.empty()

        try:
            # Call streaming endpoint
            with requests.post(
                f"{API_URL}/chat/stream",
                json={
                    "message": user_input,
                    "history": st.session_state.messages[-6:],
                    "use_rag": use_rag
                },
                stream=True,
                timeout=300
            ) as resp:
                resp.raise_for_status()

                for line in resp.iter_lines():
                    if not line:
                        continue

                    line_str = line.decode("utf-8")
                    if not line_str.startswith("data: "):
                        continue

                    data_str = line_str[6:]   # strip "data: "
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if data.get("done"):
                        break

                    token         = data.get("token", "")
                    full_response += token

                    # Update placeholder with current text + cursor
                    placeholder.markdown(full_response + "▋")

            # Final render without cursor
            placeholder.markdown(full_response)

            # Fetch sources separately
            sources = []
            if use_rag:
                try:
                    src_resp = requests.post(
                        f"{API_URL}/chat/context",
                        json={"message": user_input, "history": []},
                        timeout=30
                    )
                    sources = src_resp.json().get("chunks", [])
                    if sources:
                        with st.expander("Sources", expanded=False):
                            for s in sources:
                                st.caption(
                                    f"[{s['source']}] similarity: {s['similarity']}"
                                )
                                st.text(s["text"][:150])
                except Exception:
                    pass

        except requests.exceptions.ConnectionError:
            placeholder.error("Cannot connect to API. Run stream_api.py first.")
            full_response = "Error: API not running."

        # Save to history
        st.session_state.messages.append({
            "role":    "assistant",
            "content": full_response,
            "sources": sources
        })