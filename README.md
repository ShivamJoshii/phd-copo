# CO-PO Pairwise Mapping System

This repository contains a practical baseline implementation of the **CO-PO pairwise mapping framework** for Outcome-Based Education (OBE).

## What it does

Given a list of Course Outcomes (COs) and Program Outcomes (POs), the system:

1. Builds the full Cartesian product of CO-PO pairs.
2. Extracts educational features (action intent, Bloom level, domain overlap).
3. Computes semantic similarity (TF-IDF cosine).
4. Predicts mapping strength on a 4-point scale (`0,1,2,3`).
5. Exports pairwise predictions and a matrix view.

## Project status

This is an initial MVP that focuses on a transparent, explainable baseline architecture with clear extension points for:

- SBERT embeddings
- cross-encoder scoring
- XGBoost-based final classification
- faculty-in-the-loop review workflows

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Run pairwise prediction (rule-based baseline)

```bash
copo-map \
  --co-file examples/co.json \
  --po-file examples/po.json \
  --out-dir outputs
```

Outputs:

- `outputs/pair_predictions.csv`
- `outputs/matrix.csv`

## Input format

`co.json`:

```json
[
  {"id": "CO1", "text": "Design and implement relational database solutions."},
  {"id": "CO2", "text": "Analyze algorithmic efficiency for real-world problems."}
]
```

`po.json`:

```json
[
  {"id": "PO1", "text": "Identify, formulate, and solve complex engineering problems."},
  {"id": "PO2", "text": "Design solutions that meet specified needs."}
]
```

## Architecture mapping to specification

- **Layer 1**: preprocessing (`copo_mapper/preprocess.py`)
- **Layer 2**: structural extraction (`copo_mapper/features.py`)
- **Layer 3**: semantic representation (`copo_mapper/semantic.py`)
- **Layer 4/5**: pair scoring and educational features (`copo_mapper/scoring.py`)
- **Layer 6**: final 4-class decision (`copo_mapper/scoring.py`)

## Next milestones

1. Add sentence-transformer embeddings.
2. Add cross-encoder pair scorer.
3. Add trainable XGBoost classifier on labeled faculty data.
4. Build review UI/API for human corrections and feedback loop.




## Streamlit UI

Launch the browser UI with:

```bash
pip install streamlit
streamlit run streamlit_app.py
```

Then in the app:
1. Upload `CO` JSON and `PO` JSON files.
2. Click **Run Mapping**.
3. Inspect the color-coded CO-PO matrix.
4. Select a CO and PO to view detailed prediction info.
5. Use export buttons to download `pair_predictions.csv` and `matrix.csv`.

## How to test

### 1) Fast smoke test (no installs)

```bash
python -m compileall copo_mapper
python -m copo_mapper.cli --co-file examples/co.json --po-file examples/po.json --out-dir outputs
```

You should see:

- `Saved pair predictions: outputs/pair_predictions.csv`
- `Saved matrix: outputs/matrix.csv`

### 2) Run unit test

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

This executes a temporary-file smoke test that validates the end-to-end pipeline and output shape.


## Create a Pull Request

If you want to open a PR from `work` to `main`:

```bash
git checkout work
git push -u origin work
```

Then open a PR in your Git host UI with:
- **base branch:** `main`
- **compare branch:** `work`

If `main` does not exist yet, create and push it first:

```bash
git checkout -b main
git push -u origin main
git checkout work
```
