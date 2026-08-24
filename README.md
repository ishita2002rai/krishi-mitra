# Krishi-Mitra

An agentic AI crop advisory assistant for farmers. Upload a photo of a crop leaf and
the system runs a full perceive → reason → gather-context → decide → act loop: it
classifies the disease, asks a clarifying question when it isn't confident, checks
local weather to assess spread risk, and produces a treatment plan — then lets you
track whether that treatment actually worked.

Built as a final-year project using an iterative/incremental SDLC: each increment below
was built, demoed, and tested before starting the next.

## Features

| # | Increment | What it does |
|---|---|---|
| 1 | **Disease classifier** | MobileNetV2 transfer learning on PlantVillage -- 35 classes across 11 crops (apple, cherry, corn, grape, orange, peach, pepper, potato, squash, strawberry, tomato). |
| 2 | **Clarifying-questions agent** | A LangGraph loop asks one follow-up question via Groq when the classifier's top prediction is below 85% confidence, then refines the diagnosis from the answer. Also flags likely out-of-scope photos (e.g. a crop the app doesn't support) instead of silently forcing a guess. |
| 3 | **Weather-aware risk reasoning** | Fetches current + 3-day forecast weather (Open-Meteo) for the farmer's location and reasons about whether conditions will accelerate or slow disease spread. |
| 4 | **Treatment plan + app + history** | Generates an organic + chemical treatment plan (metric dosages), a Streamlit UI tying the whole loop together, and SQLite-backed scan history. |
| 5 | **Dashboard, feedback loop, PDF export, chatbot** | Farm-wide analytics (interactive charts), a 1–5 treatment-outcome rating that closes the agentic loop, plot/field tagging, per-scan PDF reports, and a general-purpose chat widget. |

## Results (Increment 1, 35-class multi-crop model)

| Metric | Value |
|---|---|
| Validation accuracy | 96.4% |
| Macro F1 | 0.967 |
| Weighted F1 | 0.965 |

Cross-crop classes are easy to tell apart (many at F1 1.00 -- an apple leaf and a corn
leaf don't look alike). Almost all the remaining confusion is *within* tomato's own
10 sub-classes, the visually similar diseases first trained on: Target Spot (F1 0.83)
and Early blight (F1 0.88) are still the hardest pair, plus Potato___healthy (F1 0.89,
only 30 validation images). Full breakdown in `outputs/metrics.json`, confusion matrix
in `outputs/confusion_matrix.png`.

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
Upload a leaf photo, enter your location, and click Diagnose. The Diagnose tab shows
which crops are supported. Below 85% confidence, or if the model's guesses disagree on
the crop species itself (a sign the photo may not be one of the supported crops),
you'll be asked one clarifying question before the final diagnosis. Non-healthy
diagnoses get a weather-based risk assessment and a treatment plan (organic + generic
chemical, metric dosages). Every scan is saved to the **History** tab, where you can
rate how the treatment worked and download a PDF report. The **Dashboard** tab
aggregates trends across all your scans. The floating chat button (bottom-right) opens
a general-purpose crop-advisory chatbot that redirects you to the right tab for
anything it can't answer directly (e.g. "analyze my photo" -> Diagnose tab) and knows
which crops are and aren't supported.

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
- **11 of PlantVillage's 14 crop species** — Blueberry, Raspberry, and Soybean are
  excluded because this dataset only has "healthy" images for them (no disease
  examples), so a model trained on them could never actually diagnose anything for
  those three crops.
- **Some crops have far less training data than others** (e.g. Potato___healthy has
  ~150 images vs. Orange's ~5,500 for its one class) — class weighting compensates
  during training, but very sparse classes will still be less reliable than
  well-represented ones like tomato.
- **Treatment plans don't distinguish disease category.** For a wood/vascular disease
  like grapevine Esca, the LLM generated a plausible-looking chemical spray regimen,
  but in real viticulture practice curative fungicides have limited effect on
  established wood infections -- management is almost entirely cultural (pruning-wound
  protection, removing infected wood). The model reasons from general agronomy
  knowledge per-diagnosis, not a disease-category-aware framework that would flag
  "this is a vascular disease, de-emphasize curative chemical treatment."
- **Out-of-scope crop detection is a heuristic, not a hard guarantee.** It flags when
  the classifier's confident candidates disagree on crop species, which catches clear
  cases, but a genuinely unfamiliar crop that happens to resemble a supported one could
  still get a wrong diagnosis with no warning.

## Development notes

A few real issues were found through actual testing (not just code review) and fixed
during development -- kept here since the debugging process is itself part of the
engineering story:

- A Dashboard chart crashed in the browser (`pd.cut()` producing `Interval` objects
  that Streamlit's chart backend can't serialize) -- fixed and the fix was verified by
  replicating Altair's exact validation call outside Streamlit before shipping it again.
- Downloaded PDF reports were full of stray "?" characters -- traced to typographic
  Unicode space variants the LLM uses that `fpdf2`'s fonts can't render; fixed generally
  via `unicodedata` category checks rather than hand-listing every space variant.
- The chat widget appeared to "lose context" after a few turns. Rather than assuming a
  session-state bug, the exact failing conversation was replayed directly against the
  LLM backend and the literal message list sent on the failing call was inspected --
  confirming the history *was* being sent correctly, and the model itself just wasn't
  using it well. Fixed with an explicit prompt reinforcement, then re-verified against
  the original failing conversation end-to-end.
