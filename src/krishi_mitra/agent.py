"""LangGraph agent loop for Increment 2: classify a leaf photo, and if the top
prediction isn't confident enough, ask one clarifying question before
committing to a diagnosis.

Two-pass usage (see scripts/run_agent.py):
    graph = build_graph()
    state = graph.invoke({"image_path": path})
    if not state.get("diagnosis"):
        state["user_answer"] = input(state["clarifying_question"] + " ")
        state = graph.invoke(state)
    print(state["diagnosis"], state["reasoning"])
"""

import os
from typing import List, Optional, TypedDict

from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from krishi_mitra.classifier import predict_topk
from krishi_mitra.config import CONFIDENCE_THRESHOLD, GROQ_MODEL, TOP_K, display_name


class AgentState(TypedDict, total=False):
    image_path: str
    top_k: List[dict]
    confidence: float
    clarifying_question: Optional[str]
    user_answer: Optional[str]
    diagnosis: Optional[str]
    reasoning: Optional[str]


def _llm() -> ChatGroq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing GROQ_API_KEY. Copy .env.example to .env and fill it in from "
            "https://console.groq.com -> API Keys."
        )
    return ChatGroq(model=GROQ_MODEL, api_key=api_key, temperature=0.2)


def classify(state: AgentState) -> dict:
    top_k = predict_topk(state["image_path"], k=TOP_K)
    return {"top_k": top_k, "confidence": top_k[0]["confidence"]}


def finalize_high_confidence(state: AgentState) -> dict:
    top = state["top_k"][0]
    return {
        "diagnosis": top["class"],
        "reasoning": (
            f"Classifier confidence {top['confidence']:.0%} is above the "
            f"{CONFIDENCE_THRESHOLD:.0%} threshold, so no clarification was needed."
        ),
    }


def ask_clarifying_question(state: AgentState) -> dict:
    candidates = "\n".join(
        f"- {display_name(c['class'])} ({c['confidence']:.0%} confidence)" for c in state["top_k"]
    )
    prompt = (
        "You are an agronomy assistant helping diagnose a tomato plant disease from a leaf "
        "photo. A CNN classifier is unsure between these candidates:\n"
        f"{candidates}\n\n"
        "Ask the farmer ONE short, simple, non-technical question (a single sentence) whose "
        "answer would best help distinguish between these specific candidates. "
        "Return only the question, nothing else."
    )
    question = _llm().invoke(prompt).content.strip()
    return {"clarifying_question": question}


def refine_with_answer(state: AgentState) -> dict:
    candidates = "\n".join(
        f"- {c['class']}: {display_name(c['class'])} ({c['confidence']:.0%} confidence)"
        for c in state["top_k"]
    )
    prompt = (
        "You are an agronomy assistant. A CNN classifier proposed these tomato disease "
        f"candidates:\n{candidates}\n\n"
        f'You asked the farmer: "{state["clarifying_question"]}"\n'
        f'They answered: "{state["user_answer"]}"\n\n'
        "Pick exactly ONE candidate as the final diagnosis using its EXACT class name from "
        "the list above (the part before the colon), and give one sentence of reasoning. "
        "Respond in exactly this format:\n"
        "DIAGNOSIS: <exact class name>\n"
        "REASONING: <one sentence>"
    )
    reply = _llm().invoke(prompt).content.strip()

    valid_classes = {c["class"] for c in state["top_k"]}
    diagnosis, reasoning = state["top_k"][0]["class"], reply
    for line in reply.splitlines():
        if line.upper().startswith("DIAGNOSIS:"):
            candidate = line.split(":", 1)[1].strip()
            if candidate in valid_classes:
                diagnosis = candidate
        if line.upper().startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip()

    return {"diagnosis": diagnosis, "reasoning": reasoning}


def _route_after_classify(state: AgentState) -> str:
    return "finalize_high_confidence" if state["confidence"] >= CONFIDENCE_THRESHOLD else "ask_clarifying_question"


def _route_entry(state: AgentState) -> str:
    return "refine_with_answer" if state.get("user_answer") else "classify"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("classify", classify)
    graph.add_node("finalize_high_confidence", finalize_high_confidence)
    graph.add_node("ask_clarifying_question", ask_clarifying_question)
    graph.add_node("refine_with_answer", refine_with_answer)

    graph.set_conditional_entry_point(
        _route_entry, {"classify": "classify", "refine_with_answer": "refine_with_answer"}
    )
    graph.add_conditional_edges(
        "classify",
        _route_after_classify,
        {
            "finalize_high_confidence": "finalize_high_confidence",
            "ask_clarifying_question": "ask_clarifying_question",
        },
    )
    graph.add_edge("finalize_high_confidence", END)
    graph.add_edge("ask_clarifying_question", END)
    graph.add_edge("refine_with_answer", END)
    return graph.compile()
