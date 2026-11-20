import streamlit as st
from typing import List, Dict, Any

from backend.core import run_llm


def _format_sources(context_docs: List[Any]) -> List[str]:
    return [
        str((meta.get("source") or "Unknown"))
        for doc in (context_docs or [])
        if (meta := (getattr(doc, "metadata", None) or {})) is not None
    ]

st.set_page_config(page_title="LangChain Documentation Helper", layout="centered")
st.title("LangChain Documentation Helper")

with st.sidebar:
    st.sidebar.header("Session")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.pop("messages", None)
        st.rerun()

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "context": "Ask me anything about LangChain docs. I'll retrieve relevant context and cite sources.",
            "sources": ["www.google.com"]
        }
    ]

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["context"])
        if msg["sources"]:
            with st.expander("Sources"):
                for source in msg["sources"]:
                    st.markdown(f"- {source}")

prompt = st.chat_input("Ask question about LangChain & LangGraph...")
if prompt:
    st.session_state["messages"].append({"role": "assistant", "context": prompt, "sources":[]})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving docs and generating answer..."):
                result: Dict[str, Any] = run_llm(prompt)
                if result["answer"]:
                    answer = result["answer"]
                    sources = result["context"]
                    st.session_state["messages"].append({"role": "assistant", "context": answer,
                                                         "sources": _format_sources(sources or [])})
                    st.markdown(answer)
                    if sources:
                            with st.expander("Source"):
                                for source in sources:
                                    st.markdown(f"- {source.metadata['source'] or 'Unknown'}")
        except Exception as e:
            st.error("Failed to generate a response.")
            st.exception(e)