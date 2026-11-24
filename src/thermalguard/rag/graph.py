"""LangGraph maintenance assistant: conversation memory, engine-aware tool lookup, cited RAG answers.

        START -> analyze --(engine id mentioned?)--> asset_lookup -> retrieve -> generate -> END
                         \-------------------------------------> retrieve ---^

Memory: MemorySaver checkpointer keyed by thread_id (one per Streamlit session).
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Annotated, TypedDict

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from thermalguard import config
from thermalguard.rag.index import Retriever
from thermalguard.rag.llm import get_llm

ASSET_RE = re.compile(r"\b(FD00[1-4])[_\s-]*(train|test)[_\s-]*(\d{1,3})\b", re.I)
SHORT_RE = re.compile(r"\b(?:engine|unit|asset)\s*#?\s*(\d{1,3})\b", re.I)

SYSTEM = """You are ThermalGuard, a maintenance assistant for plant operators.
Answer ONLY from the numbered sources and the live engine facts below. If they do not contain the answer,
say you don't know. Cite sources inline with plain ASCII brackets like [1], [2] (never 【1】). Be concise and practical; give the required action
when the question is about an engine's condition. Predictions are decision support, not a removal order."""


class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    question: str
    asset_id: str | None
    asset_facts: dict | None
    docs: list[dict]


@lru_cache(maxsize=1)
def _retriever() -> Retriever:
    return Retriever()


def _fleet_status() -> pd.DataFrame | None:
    p = config.ARTIFACTS_DIR / "fleet" / "status.parquet"
    return pd.read_parquet(p) if p.exists() else None


def find_asset(text: str) -> str | None:
    if m := ASSET_RE.search(text):
        return f"{m.group(1).upper()}_{m.group(2).lower()}_{int(m.group(3)):03d}"
    if m := SHORT_RE.search(text):  # "engine 31" -> default fleet (FD001 test engines)
        return f"FD001_test_{int(m.group(1)):03d}"
    return None


def analyze(state: State) -> State:
    humans = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    q = humans[-1].content
    asset = find_asset(q) or state.get("asset_id")  # follow-ups keep the engine in focus
    if len(humans) > 1:  # condense follow-ups into a standalone question using the chat history
        history = "\n".join(f"{type(m).__name__[:-7]}: {m.content}" for m in state["messages"][-6:-1])
        q = get_llm().invoke(
            f"Rewrite the final user question as a standalone question using the conversation.\n"
            f"Conversation:\n{history}\nFinal question: {q}\nStandalone question:").content.strip()
    return {"question": q, "asset_id": asset, "asset_facts": None}


def asset_lookup(state: State) -> State:
    fleet = _fleet_status()
    if fleet is None or state["asset_id"] not in set(fleet.asset_id):
        return {"asset_facts": {"asset_id": state["asset_id"], "error": "engine not found in fleet status"}}
    r = fleet[fleet.asset_id == state["asset_id"]].iloc[0]
    facts = {"asset_id": r.asset_id, "last_cycle": int(r.last_cycle), "predicted_rul_cycles": float(r.rul_pred),
             "status": r.status}
    if "first_anomaly_cycle" in r and pd.notna(r.first_anomaly_cycle):
        facts["first_anomaly_alert_cycle"] = int(r.first_anomaly_cycle)
    return {"asset_facts": facts}


def retrieve(state: State) -> State:
    return {"docs": _retriever().search(state["question"], k=5, asset_id=state.get("asset_id"))}


def generate(state: State) -> State:
    ctx = "\n\n".join(f"[{i}] ({d['source']})\n{d['text']}" for i, d in enumerate(state["docs"], 1))
    facts = state.get("asset_facts")
    facts_txt = f"\nLive engine facts (from ThermalGuard models): {facts}\n" if facts else ""
    msgs = [SystemMessage(SYSTEM + facts_txt + "\nSources:\n" + ctx),
            *state["messages"][-6:-1], HumanMessage(state["question"])]
    answer = get_llm().invoke(msgs).content
    return {"messages": [AIMessage(answer, additional_kwargs={
        "citations": [{"n": i, "source": d["source"], "text": d["text"]} for i, d in enumerate(state["docs"], 1)],
        "asset_facts": facts})]}


def route(state: State) -> str:
    return "asset_lookup" if state.get("asset_id") else "retrieve"


@lru_cache(maxsize=1)
def build_graph():
    g = StateGraph(State)
    for name, fn in [("analyze", analyze), ("asset_lookup", asset_lookup), ("retrieve", retrieve),
                     ("generate", generate)]:
        g.add_node(name, fn)
    g.add_edge(START, "analyze")
    g.add_conditional_edges("analyze", route, ["asset_lookup", "retrieve"])
    g.add_edge("asset_lookup", "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile(checkpointer=MemorySaver())


def ask(question: str, thread_id: str = "cli") -> AIMessage:
    out = build_graph().invoke({"messages": [HumanMessage(question)]},
                               config={"configurable": {"thread_id": thread_id}})
    return out["messages"][-1]


if __name__ == "__main__":
    import sys

    msg = ask(" ".join(sys.argv[1:]) or "What does a warning status mean and what should I do?")
    print(msg.content)
    for c in msg.additional_kwargs["citations"]:
        print(f"  [{c['n']}] {c['source']}")
