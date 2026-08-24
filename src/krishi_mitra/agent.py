"""LangGraph agent loop: classify a leaf photo, ask one clarifying question if
the top prediction isn't confident enough (Increment 2), then -- for any
non-healthy diagnosis -- fetch local weather, reason about disease-spread risk
(Increment 3), and generate a treatment plan (Increment 4).

Two-pass usage (see scripts/run_agent.py and app.py):
    graph = build_graph()
    state = graph.invoke({"image_path": path, "location": "Pune, India"})
    if not state.get("diagnosis"):
        state["user_answer"] = input(state["clarifying_question"] + " ")
        state = graph.invoke(state)
    print(state["diagnosis"], state["reasoning"])
    if state.get("risk_assessment"):
        print(state["risk_assessment"])
    if state.get("treatment_plan"):
        print(state["treatment_plan"])
"""

import os
from typing import List, Optional, TypedDict

from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from krishi_mitra.classifier import predict_topk
from krishi_mitra.config import (
    CONFIDENCE_THRESHOLD,
    GROQ_MODEL,
    SUPPORTED_CROPS,
    TOP_K,
    TREATMENT_DISCLAIMER,
    crop_name,
    display_name,
)
from krishi_mitra.weather import fetch_weather_summary, format_summary


class AgentState(TypedDict, total=False):
    image_path: str
    location: str
    top_k: List[dict]
    confidence: float
    clarifying_question: Optional[str]
    user_answer: Optional[str]
    diagnosis: Optional[str]
    reasoning: Optional[str]
    weather: Optional[dict]
    risk_assessment: Optional[str]
    treatment_plan: Optional[str]


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


CROP_DISAGREEMENT_THRESHOLD = 0.05  # ignore near-zero tail candidates when comparing crops


def _spans_multiple_crops(top_k) -> bool:
    """True if the candidates with non-trivial probability disagree on the crop
    species itself, not just the disease -- a stronger sign of a possibly-
    unsupported/out-of-scope photo than low confidence alone, since a softmax
    classifier can be confidently wrong on inputs unlike anything it was trained on.
    Restricted to candidates above CROP_DISAGREEMENT_THRESHOLD: with 35 classes, a
    dominant top-1 (e.g. 99.99%) still leaves ~0.01% spread arbitrarily across the
    other 34 classes, which can span unrelated crops by pure noise, not genuine
    ambiguity -- counting those would false-flag confident, correct diagnoses."""
    meaningful = [c for c in top_k if c["confidence"] >= CROP_DISAGREEMENT_THRESHOLD] or top_k[:1]
    return len({crop_name(c["class"]) for c in meaningful}) > 1


def ask_clarifying_question(state: AgentState) -> dict:
    top_k = state["top_k"]
    if _spans_multiple_crops(top_k):
        guessed_crops = ", ".join(sorted({crop_name(c["class"]) for c in top_k}))
        supported = ", ".join(SUPPORTED_CROPS)
        return {
            "clarifying_question": (
                f"This photo doesn't clearly match one crop -- our best guesses span different "
                f"plants ({guessed_crops}). This app currently supports: {supported}. "
                "Can you confirm what crop this is?"
            )
        }

    candidates = "\n".join(
        f"- {display_name(c['class'])} ({c['confidence']:.0%} confidence)" for c in state["top_k"]
    )
    prompt = (
        "You are an agronomy assistant helping diagnose a plant disease from a leaf "
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
        "You are an agronomy assistant. A CNN classifier proposed these plant disease "
        f"candidates:\n{candidates}\n\n"
        f'You asked the farmer: "{state["clarifying_question"]}"\n'
        f'They answered: "{state["user_answer"]}"\n\n'
        "Pick exactly ONE candidate as the final diagnosis using its EXACT class name from "
        "the list above (the part before the colon) -- even if the farmer's answer suggests "
        "their crop isn't one of these candidates or isn't supported by this app "
        f"(supported crops: {', '.join(SUPPORTED_CROPS)}), since that's the closest available "
        "guess; in that case, clearly say in the reasoning that the crop may not be supported "
        "and the diagnosis is unreliable. Otherwise, base your reasoning ONLY on symptoms the "
        "farmer actually stated -- never claim they confirmed or described something they did "
        "not mention. If their answer does not clearly address the question, say so in the "
        "reasoning and fall back to the classifier's highest-confidence candidate. "
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


def gather_weather(state: AgentState) -> dict:
    weather = fetch_weather_summary(state["location"])
    return {"weather": weather}


def assess_risk(state: AgentState) -> dict:
    prompt = (
        f"You are an agronomy assistant. A {crop_name(state['diagnosis'])} plant was diagnosed with "
        f"{display_name(state['diagnosis'])}.\n\n"
        f"{format_summary(state['weather'])}\n\n"
        "In 2-3 sentences, assess whether these current and upcoming weather conditions are "
        "likely to accelerate or slow the spread of this disease, and give one piece of "
        "immediate monitoring advice. Do not give a full treatment plan."
    )
    risk_assessment = _llm().invoke(prompt).content.strip()
    return {"risk_assessment": risk_assessment}


def generate_treatment_plan(state: AgentState) -> dict:
    prompt = (
        f"You are an agronomy assistant. A {crop_name(state['diagnosis'])} plant was diagnosed with "
        f"{display_name(state['diagnosis'])}.\n\n"
        f"Risk assessment: {state['risk_assessment']}\n\n"
        "Write a short treatment plan for the farmer with two sections:\n"
        "1. Organic/home remedies\n"
        "2. Chemical treatment (generic active ingredient names, not brand names, with "
        "general dosage/frequency guidance)\n"
        "This is for a farmer in India: use metric units only (ml or g per litre of water, "
        "or kg/litres per acre or hectare), never imperial units (no gallons, quarts, or pounds). "
        "Keep each section to 2-3 concise bullet points. Do not include any other sections."
    )
    plan = _llm().invoke(prompt).content.strip()
    return {"treatment_plan": f"{plan}\n\n---\n{TREATMENT_DISCLAIMER}"}


def _route_after_classify(state: AgentState) -> str:
    confident_and_consistent = state["confidence"] >= CONFIDENCE_THRESHOLD and not _spans_multiple_crops(
        state["top_k"]
    )
    return "finalize_high_confidence" if confident_and_consistent else "ask_clarifying_question"


def _route_entry(state: AgentState) -> str:
    return "refine_with_answer" if state.get("user_answer") else "classify"


def _route_after_diagnosis(state: AgentState) -> str:
    return "skip" if "healthy" in state["diagnosis"].lower() else "gather_weather"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("classify", classify)
    graph.add_node("finalize_high_confidence", finalize_high_confidence)
    graph.add_node("ask_clarifying_question", ask_clarifying_question)
    graph.add_node("refine_with_answer", refine_with_answer)
    graph.add_node("gather_weather", gather_weather)
    graph.add_node("assess_risk", assess_risk)
    graph.add_node("generate_treatment_plan", generate_treatment_plan)

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
    graph.add_conditional_edges(
        "finalize_high_confidence", _route_after_diagnosis, {"gather_weather": "gather_weather", "skip": END}
    )
    graph.add_conditional_edges(
        "refine_with_answer", _route_after_diagnosis, {"gather_weather": "gather_weather", "skip": END}
    )
    graph.add_edge("ask_clarifying_question", END)
    graph.add_edge("gather_weather", "assess_risk")
    graph.add_edge("assess_risk", "generate_treatment_plan")
    graph.add_edge("generate_treatment_plan", END)
    return graph.compile()
