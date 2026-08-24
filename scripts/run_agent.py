"""CLI demo for the agent loop: classify a tomato leaf photo, ask one
clarifying question if the classifier isn't confident (Increment 2), then --
for any non-healthy diagnosis -- fetch local weather, assess disease-spread
risk (Increment 3), and generate a treatment plan (Increment 4).

Usage:
    python scripts/run_agent.py path/to/leaf.jpg "Pune, India"
"""

import argparse

from dotenv import load_dotenv

from krishi_mitra.agent import build_graph
from krishi_mitra.config import display_name
from krishi_mitra.weather import format_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_path", help="Path to a tomato leaf photo")
    parser.add_argument("location", help="Farm location for weather lookup, e.g. 'Pune, India'")
    args = parser.parse_args()

    load_dotenv()
    graph = build_graph()

    state = graph.invoke({"image_path": args.image_path, "location": args.location})

    print("\nTop candidates:")
    for c in state["top_k"]:
        print(f"  {display_name(c['class'])}: {c['confidence']:.1%}")

    if not state.get("diagnosis"):
        print(f"\nClarifying question: {state['clarifying_question']}")
        state["user_answer"] = input("Your answer: ")
        state = graph.invoke(state)

    print(f"\nDiagnosis: {display_name(state['diagnosis'])}")
    print(f"Reasoning: {state['reasoning']}")

    if state.get("weather"):
        print(f"\n{format_summary(state['weather'])}")
        print(f"\nRisk assessment: {state['risk_assessment']}")

    if state.get("treatment_plan"):
        print(f"\nTreatment plan:\n{state['treatment_plan']}")


if __name__ == "__main__":
    main()
