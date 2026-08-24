# Krishi-Mitra

An agentic AI crop advisory assistant for farmers. Upload a photo of a tomato leaf and
the system runs a full perceive → reason → gather-context → decide → act loop: it
classifies the disease, asks a clarifying question when it isn't confident, checks
local weather to assess spread risk, and produces a treatment plan — then lets you
track whether that treatment actually worked.

Built as a final-year project using an iterative/incremental SDLC: each increment below
was built, demoed, and tested before starting the next.

## Features

| # | Increment | What it does |
|---|---|---|
| 1 | **Disease classifier** | MobileNetV2 transfer learning on PlantVillage (10 tomato classes). 92.1% validation accuracy. |
| 2 | **Clarifying-questions agent** | A LangGraph loop asks one follow-up question via Groq when the classifier's top prediction is below 85% confidence, then refines the diagnosis from the answer. |
| 3 | **Weather-aware risk reasoning** | Fetches current + 3-day forecast weather (Open-Meteo) for the farmer's location and reasons about whether conditions will accelerate or slow disease spread. |
| 4 | **Treatment plan + app + history** | Generates an organic + chemical treatment plan (metric dosages), a Streamlit UI tying the whole loop together, and SQLite-backed scan history. |
| 5 | **Dashboard, feedback loop, PDF export, chatbot** | Farm-wide analytics (interactive charts), a 1–5 treatment-outcome rating that closes the agentic loop, plot/field tagging, per-scan PDF reports, and a general-purpose chat widget. |

## Results (Increment 1)

| Metric | Value |
|---|---|
| Validation accuracy | 92.1% |
| Macro F1 | 0.906 |
| Weighted F1 | 0.921 |

Per-class F1 ranges from 0.82 (Early blight, Target Spot — the visually similar,
harder-to-separate classes) to 0.98 (Yellow Leaf Curl Virus). Full breakdown in
`outputs/metrics.json`, confusion matrix in `outputs/confusion_matrix.png`.

## Tech stack

- **Classifier:** TensorFlow/Keras, MobileNetV2 transfer learning
- **Agent orchestration:** LangGraph
- **LLM:** Groq (`openai/gpt-oss-120b`), free tier
- **Weather:** Open-Meteo (free, keyless)
- **Frontend:** Streamlit
- **Storage:** SQLite
- No paid APIs or hosting anywhere in the stack.

## Project structure

```
app.py                 Streamlit frontend (Increments 4-5)
scripts/
  prepare_data.py       Download PlantVillage (Kaggle), split into train/val
  train_classifier.py   Train + evaluate the MobileNetV2 classifier
  run_agent.py           CLI demo of the agent loop (Increments 2-4)
src/krishi_mitra/
  config.py              Shared paths/constants
  classifier.py          Load the trained model, run predictions
  agent.py                LangGraph state machine (classify -> clarify -> weather -> risk -> treatment)
  weather.py              Open-Meteo geocoding + forecast
  db.py                   SQLite scan history
  report.py               Per-scan PDF export
  chat.py                 General-purpose chat assistant
data/, models/, outputs/, db/    generated artifacts (gitignored)
```

## Setup

```bash
# 1. Python 3.11 virtualenv (TensorFlow compatibility)
pyenv install 3.11.15
pyenv virtualenv 3.11.15 krishi-mitra   # or: ~/.pyenv/versions/3.11.15/bin/python -m venv .venv
pyenv local krishi-mitra                 # or: source .venv/bin/activate

# 2. Install dependencies + this package (editable, so `krishi_mitra` imports work everywhere)
pip install -r requirements.txt
pip install -e .

# 3. Credentials -- copy the template and fill in with a text editor (never share these in chat/logs)
cp .env.example .env
#   KAGGLE_API_TOKEN  from https://www.kaggle.com/settings -> API -> Create New Token
#   GROQ_API_KEY      from https://console.groq.com -> API Keys (free)
#   (Open-Meteo needs no key)

# 4. Get the data and train the classifier
python scripts/prepare_data.py
python scripts/train_classifier.py
```

## Usage

**CLI** (Increments 2-4, no browser needed):
```bash
python scripts/run_agent.py path/to/leaf_photo.jpg "Pune, India"
```

**Full app** (all 5 increments):
```bash
streamlit run app.py
```
Upload a leaf photo, enter your location, and click Diagnose. Below 85% confidence
you'll be asked one clarifying question before the final diagnosis. Non-healthy
diagnoses get a weather-based risk assessment and a treatment plan (organic + generic
chemical, metric dosages). Every scan is saved to the **History** tab, where you can
rate how the treatment worked and download a PDF report. The **Dashboard** tab
aggregates trends across all your scans. The floating chat button (bottom-right) opens
a general-purpose crop-advisory chatbot that redirects you to the right tab for
anything it can't answer directly (e.g. "analyze my photo" -> Diagnose tab).

## Known limitations

Found through real testing, not just theoretical — kept here rather than hidden, since
they're legitimate discussion points for an evaluation:

- **Visually-similar diseases can still get confused.** Early blight's textbook
  "bullseye" lesion description occasionally gets matched to the literally-named
  "Target Spot" class by the LLM refinement step, since it reasons from disease names
  and general knowledge rather than a grounded symptom database.
- **The LLM can misattribute a farmer's answer** if it doesn't address the specific
  clarifying question asked — the prompt now explicitly requires it to say so and fall
  back to the classifier's top candidate rather than fabricate a match, but this relies
  on the model following that instruction, not a hard constraint.
- **Tomato-only, 10 PlantVillage classes** — deliberately scoped down per the original
  brief; extending to other crops would need retraining on a different dataset.
