"""Maintenance assistant chat (LangGraph RAG): conversation memory + source citations + live engine facts."""
import uuid

import streamlit as st

from _shared import ART, need, page_header

st.set_page_config(page_title="Assistant · ThermalGuard", page_icon=":material/forum:", layout="wide")
page_header("LangGraph RAG", "Maintenance assistant",
            "Answers from the maintenance manual, NASA documentation and maintenance logs, with cited sources. "
            "Mention an engine (for example <i>engine 31</i>) to pull its live RUL and anomaly status.")

from thermalguard.rag.llm import KEY_VARS, llm_available, provider  # noqa: E402

if not llm_available():
    st.error(f"No LLM key configured. Add `LLM_PROVIDER={provider()}` and `{KEY_VARS.get(provider(), 'GROQ_API_KEY')}=...` "
             "to `.env` (free tier), then restart.")
    st.stop()
if not need(ART / "rag/vectors.npy", "python -m thermalguard.rag.index"):
    st.stop()

from thermalguard.rag.graph import ask  # noqa: E402

if "thread_id" not in st.session_state:
    st.session_state.thread_id, st.session_state.chat = str(uuid.uuid4()), []

with st.sidebar:
    if st.button("New conversation", icon=":material/refresh:", width="stretch"):
        st.session_state.thread_id, st.session_state.chat = str(uuid.uuid4()), []
    st.markdown("**Try asking**\n- What does a warning status mean?\n- How is engine 31 doing?\n- What should I "
                "inspect next? *(follow-up)*\n- How do I tell a sensor fault from degradation?")

AVATAR = {"user": ":material/person:", "assistant": ":material/engineering:"}


def render(turn: dict) -> None:
    with st.chat_message(turn["role"], avatar=AVATAR[turn["role"]]):
        st.markdown(turn["content"])
        if turn.get("facts"):
            st.caption(f"Live engine facts: {turn['facts']}")
        if turn.get("citations"):
            with st.expander(f"Sources ({len(turn['citations'])})"):
                for c in turn["citations"]:
                    st.markdown(f"**[{c['n']}] {c['source']}**")
                    st.caption(c["text"][:600] + ("…" if len(c["text"]) > 600 else ""))


for turn in st.session_state.chat:
    render(turn)

MAX_QUESTIONS = 15  # per visitor session: keeps a public demo inside the free LLM quota
asked = sum(t["role"] == "user" for t in st.session_state.chat)
if asked >= MAX_QUESTIONS:
    st.info(f"Demo limit of {MAX_QUESTIONS} questions per session reached (free-tier LLM). Thanks for trying it!")
    st.stop()

if q := st.chat_input("Ask about an engine, a procedure or a sensor…"):
    st.session_state.chat.append({"role": "user", "content": q})
    render(st.session_state.chat[-1])
    with st.spinner("Thinking…"):
        try:
            msg = ask(q, thread_id=st.session_state.thread_id)
            turn = {"role": "assistant", "content": msg.content,
                    "citations": msg.additional_kwargs.get("citations"), "facts": msg.additional_kwargs.get("asset_facts")}
        except Exception as e:  # free-tier rate limits etc.
            turn = {"role": "assistant", "content": f"The assistant could not answer right now: `{type(e).__name__}: {e}`"}
    st.session_state.chat.append(turn)
    render(turn)
