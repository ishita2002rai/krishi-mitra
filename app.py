"""Streamlit frontend for Krishi-Mitra: upload a leaf photo, get a diagnosis, a
weather-aware risk assessment, and a treatment plan (Increment 4); tag scans by
plot, rate treatment outcomes, browse a farm-wide analytics dashboard, and
export a PDF report per scan (Increment 5).

Usage:
    streamlit run app.py
"""

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from streamlit_searchbox import st_searchbox

from krishi_mitra.agent import build_graph
from krishi_mitra.chat import get_chat_reply
from krishi_mitra.config import display_name
from krishi_mitra.db import get_history, save_image, save_scan, update_outcome
from krishi_mitra.report import build_scan_report_pdf
from krishi_mitra.weather import format_place, format_summary, search_locations

load_dotenv()
st.set_page_config(page_title="Krishi-Mitra", page_icon="\U0001f331")

OUTCOME_LABELS = {1: "Much worse", 2: "Worse", 3: "No change", 4: "Improved", 5: "Fully resolved"}


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


def _pdf_download(scan: dict, key: str):
    try:
        pdf_bytes = build_scan_report_pdf(scan)
        st.download_button(
            "Download PDF report", data=pdf_bytes, file_name=f"krishi_mitra_{key}.pdf",
            mime="application/pdf", key=f"pdf_{key}",
        )
    except Exception as e:
        st.caption(f"Could not generate PDF: {e}")


def render_result(state, plot_name=None):
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
            scan_id = save_scan(state, plot_name=plot_name)
            st.session_state["scan_saved"] = scan_id
            st.caption(f"Saved to scan history ({scan_id[:8]}).")
        except Exception as e:
            st.caption(f"Could not save to scan history: {e}")

    location = state.get("location")
    scan_id = st.session_state.get("scan_saved", "scan")
    _pdf_download(
        {
            "location": format_place(location) if isinstance(location, dict) else location,
            "plot_name": plot_name,
            "image_path": state.get("image_path"),
            "diagnosis": state["diagnosis"],
            "confidence": state.get("confidence"),
            "reasoning": state.get("reasoning"),
            "risk_assessment": state.get("risk_assessment"),
            "treatment_plan": state.get("treatment_plan"),
        },
        key=scan_id[:8],
    )


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
    plot_name = st.text_input("Plot/field name (optional)", key="plot_name_input") or None

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
        render_result(state, plot_name=plot_name)
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

    render_result(state, plot_name=plot_name)


def history_tab():
    history = get_history()
    if not history:
        st.write("No scans yet.")
        return
    for scan in history:
        plot_suffix = f" [{scan['plot_name']}]" if scan.get("plot_name") else ""
        label = f"{scan['created_at']} -- {display_name(scan['diagnosis'])}{plot_suffix} ({scan['location'] or 'unknown location'})"
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

            if scan["treatment_plan"]:
                st.divider()
                if scan["outcome_rating"]:
                    note = f" -- {scan['outcome_note']}" if scan["outcome_note"] else ""
                    st.write(f"**Outcome:** {OUTCOME_LABELS[scan['outcome_rating']]}{note}")
                else:
                    rating = st.select_slider(
                        "Did the treatment work?",
                        options=[1, 2, 3, 4, 5],
                        value=3,
                        format_func=lambda r: OUTCOME_LABELS[r],
                        key=f"rating_{scan['id']}",
                    )
                    note = st.text_input("Notes (optional)", key=f"note_{scan['id']}")
                    if st.button("Save feedback", key=f"save_{scan['id']}"):
                        update_outcome(scan["id"], rating, note)
                        st.rerun()

            _pdf_download(scan, key=scan["id"][:8])


def _interactive_bar(counts: pd.Series, label_title: str, value_title: str = "Count", sort=None):
    """Bar chart with hover tooltips and a highlight-on-hover effect."""
    df = counts.rename(value_title).reset_index()
    df.columns = [label_title, value_title]
    hover = alt.selection_point(fields=[label_title], on="mouseover", empty=False)
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(f"{label_title}:N", sort=sort, title=None),
            y=alt.Y(f"{value_title}:Q", title=None),
            tooltip=[alt.Tooltip(f"{label_title}:N"), alt.Tooltip(f"{value_title}:Q")],
            opacity=alt.condition(hover, alt.value(1.0), alt.value(0.75)),
            color=alt.condition(hover, alt.value("#1B5E20"), alt.value("#2E7D32")),
        )
        .add_params(hover)
        .properties(height=280)
    )
    st.altair_chart(chart, use_container_width=True)


def _interactive_line(counts: pd.Series, x_title: str, value_title: str = "Count"):
    """Line chart with hover tooltips and mouse-wheel zoom / drag-to-pan."""
    df = counts.rename(value_title).reset_index()
    df.columns = [x_title, value_title]
    chart = (
        alt.Chart(df)
        .mark_line(point=True, color="#2E7D32")
        .encode(
            x=alt.X(f"{x_title}:T", title=None),
            y=alt.Y(f"{value_title}:Q", title=None),
            tooltip=[alt.Tooltip(f"{x_title}:T"), alt.Tooltip(f"{value_title}:Q")],
        )
        .properties(height=280)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


def dashboard_tab():
    history = get_history()
    if not history:
        st.write("No scans yet -- diagnose a leaf first to see analytics here.")
        return

    df = pd.DataFrame(history)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["date"] = df["created_at"].dt.normalize()
    df["diagnosis_display"] = df["diagnosis"].apply(display_name)

    plot_names = sorted(df["plot_name"].dropna().unique())
    if plot_names:
        selected_plot = st.selectbox("Show a specific field/plot", ["All fields"] + list(plot_names))
        if selected_plot != "All fields":
            df = df[df["plot_name"] == selected_plot]

    col1, col2 = st.columns(2)
    col1.metric("Crops checked", len(df))
    mode = df["diagnosis_display"].mode()
    col2.metric("Most common problem", mode.iloc[0] if not mode.empty else "-")

    st.subheader("How often each problem showed up")
    _interactive_bar(df["diagnosis_display"].value_counts(), label_title="Diagnosis", sort="-y")

    st.subheader("When you've been checking your crops")
    _interactive_line(df.groupby("date").size().rename("checks"), x_title="date", value_title="checks")

    st.subheader("How sure was the app in its diagnosis?")
    confidence = df["confidence"].dropna()
    if not confidence.empty:
        bin_edges = [0, 0.5, 0.7, 0.85, 0.95, 1.0]
        bin_labels = ["Not very sure", "A bit unsure", "Fairly sure", "Quite sure", "Very sure"]
        buckets = pd.cut(confidence, bins=bin_edges, labels=bin_labels, include_lowest=True)
        counts = buckets.value_counts().reindex(bin_labels, fill_value=0)
        _interactive_bar(counts, label_title="Confidence", value_title="checks", sort=bin_labels)
        st.caption(f"On average, the app was {confidence.mean():.0%} sure about its diagnosis, across {len(confidence)} checks.")

    st.subheader("Did the treatments work?")
    rated = df["outcome_rating"].dropna()
    if rated.empty:
        st.caption("No feedback yet -- let us know how a treatment worked in the History tab.")
    else:
        counts = rated.map(OUTCOME_LABELS).value_counts().reindex(OUTCOME_LABELS.values(), fill_value=0)
        _interactive_bar(counts, label_title="Outcome", value_title="checks", sort=list(OUTCOME_LABELS.values()))
        st.caption(f"On average, treatments were rated {rated.mean():.1f} out of 5, based on {len(rated)} check-ins.")


def chat_widget():
    st.markdown(
        """<style>
        .st-key-chat_widget { position: fixed; bottom: 20px; right: 20px; z-index: 999; width: auto; }
        .st-key-chat_widget button {
            border-radius: 24px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            transition: transform 0.15s ease, box-shadow 0.15s ease, background-color 0.15s ease;
        }
        .st-key-chat_widget button:hover {
            transform: scale(1.05);
            box-shadow: 0 6px 18px rgba(0,0,0,0.25);
            background-color: #2E7D32;
            color: white;
            border-color: #2E7D32;
        }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            background-color: #DCF8C6; border-radius: 12px;
        }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
            background-color: #FFFFFF; border: 1px solid #E0E0E0; border-radius: 12px;
        }
        </style>""",
        unsafe_allow_html=True,
    )
    with st.container(key="chat_widget"):
        with st.popover("\U0001f4ac Ask Krishi-Mitra"):
            st.caption("Ask general crop/farming questions -- not tied to a specific scan.")
            messages = st.session_state.setdefault("chat_messages", [])

            # Fixed-height scrollable message pane, auto-scrolls to the newest message.
            # st.chat_input below is placed OUTSIDE this container so it stays pinned
            # as a bottom input bar instead of drifting as the conversation grows.
            chat_area = st.container(height=350)
            with chat_area:
                for msg in messages:
                    with st.chat_message(msg["role"]):
                        st.write(msg["content"])

            prompt = st.chat_input("Ask about your crops...")
            if prompt:
                messages.append({"role": "user", "content": prompt})
                with chat_area:
                    with st.chat_message("user"):
                        st.write(prompt)
                    with st.chat_message("assistant"):
                        try:
                            with st.spinner("Thinking..."):
                                reply = get_chat_reply(messages[-10:])
                            st.write(reply)
                            messages.append({"role": "assistant", "content": reply})
                        except SystemExit as e:
                            st.error(str(e))
                        except Exception as e:
                            st.error(f"Something went wrong: {e}")


st.title("\U0001f331 Krishi-Mitra")
st.caption("Agentic crop advisory assistant -- upload a tomato leaf photo to get started.")

diagnose, dashboard, history = st.tabs(["Diagnose", "Dashboard", "History"])
with diagnose:
    diagnose_tab()
with dashboard:
    dashboard_tab()
with history:
    history_tab()

chat_widget()
