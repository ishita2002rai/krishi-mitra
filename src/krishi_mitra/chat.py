"""General crop-advisory chatbot for the bottom-right popup widget in app.py --
separate from agent.py's structured diagnose-and-treat graph, this is free-form
conversation for whatever the farmer wants to ask."""

import os

from langchain_groq import ChatGroq

from krishi_mitra.config import GROQ_MODEL, SUPPORTED_CROPS

SYSTEM_PROMPT = (
    "You are Krishi-Mitra's assistant, helping a farmer with general crop and farming "
    "questions. Keep answers short (2-4 sentences), practical, and in plain language. "
    "You are not a substitute for a local agricultural extension officer for anything "
    "high-stakes like chemical dosing -- point the farmer there when relevant.\n\n"
    f"The app's photo-based Diagnose tab only supports these crops: {', '.join(SUPPORTED_CROPS)}. "
    "If asked whether a different crop is supported, say clearly that it isn't yet.\n\n"
    "This app has three tabs, and you should redirect the farmer to the right one "
    "instead of guessing or apologizing when you can't do something yourself:\n"
    "- 'Diagnose' tab: upload a leaf photo (of a supported crop) to get an actual "
    "disease diagnosis, a weather-based risk assessment, and a treatment plan. You "
    "cannot see or analyze images yourself in this chat -- if the farmer asks you to "
    "look at, analyze, or diagnose a photo, do NOT just say you can't view images and "
    "stop there, always point them to this tab.\n"
    "- 'History' tab: browse past scans, see full diagnosis/treatment details for a "
    "specific past scan, rate how well a treatment worked, and download a PDF report. "
    "Point here for anything about a SPECIFIC past scan or wanting a record/report of one.\n"
    "- 'Dashboard' tab: farm-wide analytics -- diagnosis frequency, scans over time, "
    "classifier confidence distribution, and treatment outcome ratings, filterable by "
    "plot/field. Point here for questions about trends, patterns, or statistics across "
    "MULTIPLE scans (e.g. \"how many times have I had blight\", \"what's my most common "
    "disease\", \"show me trends\") -- you don't have direct access to this data yourself."
)


def _llm() -> ChatGroq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing GROQ_API_KEY. Copy .env.example to .env and fill it in from "
            "https://console.groq.com -> API Keys."
        )
    return ChatGroq(model=GROQ_MODEL, api_key=api_key, temperature=0.4)


def get_chat_reply(messages: list) -> str:
    """messages: [{"role": "user"|"assistant", "content": str}, ...] conversation so
    far (no system message -- added here). Returns the assistant's next reply.
    Deliberately general-purpose -- no injected scan/location context, so answers to
    generic questions (e.g. crop seasons) stay generic rather than silently assuming
    the farmer is asking about wherever their last scan happened to be."""
    reply = _llm().invoke([{"role": "system", "content": SYSTEM_PROMPT}, *messages])
    return reply.content.strip()
