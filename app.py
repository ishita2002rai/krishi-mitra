"""Streamlit frontend for Krishi-Mitra: upload a leaf photo, get a diagnosis, a
weather-aware risk assessment, and a treatment plan (Increment 4).

Usage:
    streamlit run app.py
"""

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from streamlit_searchbox import st_searchbox

from krishi_mitra.agent import build_graph
from krishi_mitra.config import display_name
from krishi_mitra.db import get_history, save_image, save_scan
from krishi_mitra.weather import format_place, format_summary, search_locations

load_dotenv()
st.set_page_config(page_title="Krishi-Mitra", page_icon="\U0001f331")


@st.cache_resource
def get_graph():
    return build_graph()


def reset_scan():
    for key in ("agent_state", "scan_saved"):
        st.session_state.pop(key, None)


def run_step(state, spinner_text):
    """Invoke the agent graph, returning (new_state, None) on success or
    (None, error_message) on failure -- never lets a raw traceback reach the UI."""
    try:
        with st.spinner(spinner_text):
            return get_graph().invoke(state), None
    except SystemExit as e:
        return None, str(e)
    except Exception as e:
        return None, f"Something went wrong: {e}"


def render_result(state):
    st.subheader("Top candidates")
    for c in state["top_k"]:
        st.write(f"- {display_name(c['class'])}: {c['confidence']:.1%}")

    if not state.get("diagnosis"):
        return

    st.subheader(f"Diagnosis: {display_name(state['diagnosis'])}")
    st.write(state["reasoning"])

    if state.get("weather"):
        st.subheader("Weather")
        st.text(format_summary(state["weather"]))
        st.subheader("Risk assessment")
        st.write(state["risk_assessment"])

    if state.get("treatment_plan"):
        st.subheader("Treatment plan")
        st.markdown(state["treatment_plan"])

    if not st.session_state.get("scan_saved"):
        try:
            scan_id = save_scan(state)
            st.session_state["scan_saved"] = scan_id
            st.caption(f"Saved to scan history ({scan_id[:8]}).")
        except Exception as e:
            st.caption(f"Could not save to scan history: {e}")


def _search_locations_for_box(query: str):
    if len(query.strip()) < 3:
        return []
    return [(format_place(place), place) for place in search_locations(query.strip())]


def location_picker():
    return st_searchbox(
        _search_locations_for_box,
        placeholder="Start typing a city name...",
        label="Farm location",
        key="location_searchbox",
    )


def diagnose_tab():
    uploaded = st.file_uploader("Tomato leaf photo", type=["jpg", "jpeg", "png"])
    place = location_picker()

    if st.button("Diagnose", type="primary", disabled=not (uploaded and place)):
        reset_scan()
        image_path = save_image(uploaded.getvalue(), suffix="." + uploaded.name.rsplit(".", 1)[-1])
        state, error = run_step(
            {"image_path": image_path, "location": place}, "Classifying photo and checking conditions..."
        )
        if error:
            st.error(error)
        else:
            st.session_state["agent_state"] = state

    state = st.session_state.get("agent_state")
    if state is None:
        return

    if uploaded:
        st.image(uploaded, width=300)

    if not state.get("diagnosis"):
        render_result(state)
        st.info(state["clarifying_question"])
        answer = st.text_input("Your answer", key="clarify_answer")
        if st.button("Submit answer") and answer:
            state["user_answer"] = answer
            new_state, error = run_step(state, "Thinking about your answer...")
            if error:
                st.error(error)
            else:
                st.session_state["agent_state"] = new_state
                st.rerun()
        return

    render_result(state)


def history_tab():
    history = get_history()
    if not history:
        st.write("No scans yet.")
        return
    for scan in history:
        label = f"{scan['created_at']} -- {display_name(scan['diagnosis'])} ({scan['location'] or 'unknown location'})"
        with st.expander(label):
            if scan["image_path"] and Path(scan["image_path"]).exists():
                st.image(scan["image_path"], width=200)
            if scan["confidence"] is not None:
                st.write(f"Confidence: {scan['confidence']:.1%}")
            if scan["reasoning"]:
                st.write(scan["reasoning"])
            if scan["risk_assessment"]:
                st.markdown(f"**Risk assessment:** {scan['risk_assessment']}")
            if scan["treatment_plan"]:
                st.markdown(scan["treatment_plan"])


st.title("\U0001f331 Krishi-Mitra")
st.caption("Agentic crop advisory assistant -- upload a tomato leaf photo to get started.")

diagnose, history = st.tabs(["Diagnose", "History"])
with diagnose:
    diagnose_tab()
with history:
    history_tab()
