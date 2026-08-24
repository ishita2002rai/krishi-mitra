"""CLI demo for Increment 2: classify a tomato leaf photo, ask one clarifying
question if the classifier isn't confident, then report a final diagnosis.

Usage:
    python scripts/run_agent.py path/to/leaf.jpg
"""

import argparse

from dotenv import load_dotenv

from krishi_mitra.agent import build_graph
from krishi_mitra.config import display_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_path", help="Path to a tomato leaf photo")
    args = parser.parse_args()

    load_dotenv()
    graph = build_graph()

    state = graph.invoke({"image_path": args.image_path})

    print("\nTop candidates:")
    for c in state["top_k"]:
        print(f"  {display_name(c['class'])}: {c['confidence']:.1%}")

    if not state.get("diagnosis"):
        print(f"\nClarifying question: {state['clarifying_question']}")
        state["user_answer"] = input("Your answer: ")
        state = graph.invoke(state)

    print(f"\nDiagnosis: {display_name(state['diagnosis'])}")
    print(f"Reasoning: {state['reasoning']}")


if __name__ == "__main__":
    main()
