# Krishi-Mitra

Agentic AI crop advisory assistant for farmers. Given a photo of a crop leaf,
the system classifies the disease, asks clarifying questions when confidence
is low, pulls in local weather, and produces a treatment plan.

Built iteratively/incrementally:

1. **Increment 1 (done):** MobileNetV2 transfer-learning classifier for
   tomato diseases (PlantVillage dataset), with train/val evaluation. 92.1% val accuracy.
2. **Increment 2 (done):** LangGraph agent loop — clarifying questions on low-confidence predictions.
3. **Increment 3 (done):** Add local weather context (Open-Meteo) to the agent's reasoning.
4. **Increment 4 (done):** Treatment plan generation, Streamlit frontend, SQLite scan history.

## Setup (Increment 1)

```bash
# 1. Python 3.11 virtualenv
pyenv install 3.11.15   # if not already installed
pyenv virtualenv 3.11.15 krishi-mitra   # or: ~/.pyenv/versions/3.11.15/bin/python -m venv .venv
pyenv local krishi-mitra                # or: source .venv/bin/activate

# 2. Install dependencies + this package (editable, so `krishi_mitra` imports work from scripts/)
pip install -r requirements.txt
pip install -e .

# 3. Kaggle credentials for dataset download
cp .env.example .env
# fill in KAGGLE_API_TOKEN from https://www.kaggle.com/settings -> API -> Create New Token
# (paste it into .env with a text editor -- never share this token in chat/logs)

# 4. Prepare data (downloads PlantVillage, splits tomato classes into data/processed/{train,val})
python scripts/prepare_data.py

# 5. Train + evaluate
python scripts/train_classifier.py
```

## Outputs

- `models/mobilenetv2_tomato.keras` — trained model
- `models/class_names.json` — index-to-class-name mapping
- `outputs/metrics.json` — accuracy, precision/recall/F1 (macro + weighted + per-class)
- `outputs/confusion_matrix.png`
- `outputs/training_history.png`
- `data/processed/split_counts.json` — per-class train/val image counts

## Setup (Increments 2-3)

```bash
# Get a free Groq API key from https://console.groq.com -> API Keys
# and add it to .env (paste with a text editor -- never share this key in chat/logs):
#   GROQ_API_KEY=...
# No key needed for weather -- Open-Meteo is free and keyless.

python scripts/run_agent.py path/to/leaf_photo.jpg "Pune, India"
```

The agent classifies the photo (Increment 1's model). If the top prediction is below
85% confidence, it asks one clarifying question via Groq, takes your answer on the CLI,
and gives a final diagnosis + reasoning. For any non-healthy diagnosis, it then fetches
current + 3-day forecast weather for the given location (Open-Meteo, geocoded from the
city name) and reasons about whether those conditions will accelerate or slow disease
spread, plus one piece of monitoring advice. Healthy diagnoses and high-confidence
predictions each skip their respective extra step.

## Setup (Increment 4)

```bash
streamlit run app.py
```

Opens a browser UI: upload a leaf photo, type your farm's location (live suggestions
via Open-Meteo's geocoding), and click Diagnose. Runs the full agent loop (classify ->
clarify if needed -> weather -> risk -> treatment plan) and shows each step. Every
completed scan is saved to a local SQLite DB (`db/scans.db`) and browsable in the
History tab. The treatment plan covers both organic/home remedies and generic
(non-brand) chemical options in metric dosages, with a disclaimer to confirm locally
before applying chemicals.

## Project layout

```
app.py                Streamlit frontend (Increment 4)
src/krishi_mitra/     shared config, classifier inference, agent graph, weather, db -- importable as a package
scripts/              standalone entry-point scripts (data prep, training, agent CLI)
data/raw/             downloaded dataset (gitignored)
data/processed/       train/val split used for training (gitignored)
models/               saved model + class name mapping (gitignored)
outputs/              metrics, plots (gitignored)
db/                   SQLite scan history + uploaded photos (gitignored)
```
