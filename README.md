# Krishi-Mitra

Agentic AI crop advisory assistant for farmers. Given a photo of a crop leaf,
the system classifies the disease, asks clarifying questions when confidence
is low, pulls in local weather, and produces a treatment plan.

Built iteratively/incrementally:

1. **Increment 1 (done):** MobileNetV2 transfer-learning classifier for
   tomato diseases (PlantVillage dataset), with train/val evaluation. 92.1% val accuracy.
2. **Increment 2 (done):** LangGraph agent loop — clarifying questions on low-confidence predictions.
3. **Increment 3:** Add local weather context (Open-Meteo) to the agent's reasoning.
4. **Increment 4:** Treatment plan generation (Groq LLM), Streamlit frontend, SQLite scan history.

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

## Setup (Increment 2)

```bash
# Get a free Groq API key from https://console.groq.com -> API Keys
# and add it to .env (paste with a text editor -- never share this key in chat/logs):
#   GROQ_API_KEY=...

python scripts/run_agent.py path/to/leaf_photo.jpg
```

The agent classifies the photo (Increment 1's model). If the top prediction is below
85% confidence, it asks one clarifying question via Groq, takes your answer on the CLI,
and gives a final diagnosis + reasoning. High-confidence predictions skip the
clarifying question entirely.

## Project layout

```
src/krishi_mitra/    shared config, classifier inference, agent graph -- importable as a package
scripts/             standalone entry-point scripts (data prep, training, agent CLI)
data/raw/            downloaded dataset (gitignored)
data/processed/      train/val split used for training (gitignored)
models/              saved model + class name mapping (gitignored)
outputs/             metrics, plots (gitignored)
```
