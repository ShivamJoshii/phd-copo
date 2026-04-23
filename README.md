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


### Dynamic-size example (more than 3 COs/POs)

The pipeline is fully dynamic and processes **all** CO and PO items from input JSON files.
If input has `N` COs and `M` POs, output pair rows = `N × M`.

Example with 4 COs and 5 POs:

```bash
python -m copo_mapper.cli --co-file examples/co_large.json --po-file examples/po_large.json --out-dir outputs_large
```

Expected outputs:
- `outputs_large/pair_predictions.csv` with `20` rows (4×5)
- `outputs_large/matrix.csv` with full 4x5 mapping grid


## Input format

All Stage 1 and Stage 2 input files accept **either JSON or CSV**. The parser is selected by file extension (`.json` or `.csv`).

### Stage 1

`co.json` / `co.csv` — columns **`CO`** and **`description`**.

```json
[
  {"CO": "CO1", "description": "Design and implement relational database solutions."},
  {"CO": "CO2", "description": "Analyze algorithmic efficiency for real-world problems."}
]
```

```csv
CO,description
CO1,Design and implement relational database solutions.
CO2,Analyze algorithmic efficiency for real-world problems.
```

`po.json` / `po.csv` — columns **`PO`** and **`description`**.

```json
[
  {"PO": "PO1", "description": "Identify, formulate, and solve complex engineering problems."},
  {"PO": "PO2", "description": "Design solutions that meet specified needs."}
]
```

```csv
PO,description
PO1,"Identify, formulate, and solve complex engineering problems."
PO2,Design solutions that meet specified needs.
```

### Stage 2

`co_attainment.json` / `co_attainment.csv` — columns `co_id, ma_attainment, ea_attainment, indirect_attainment`.

`attainment_config.json` / `attainment_config.csv` — keys `ma_weight, ea_weight, direct_weight, indirect_weight, co_target_level, po_target_level` (single-row CSV or JSON object).

`mapping_matrix.csv` — first column `co_id`, remaining columns are PO ids.

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




## Streamlit UI (Connected Stage 1 + Stage 2)

Launch the browser UI with:

```bash
pip install streamlit
streamlit run streamlit_app.py
```

In the same Streamlit app, you now have two tabs:
- **Stage 1: Mapping** (CO/PO upload, pair scoring, matrix view)
- **Stage 2: Attainment** (uses Stage 1 matrix automatically, or accepts uploaded matrix CSV)

So attainment is **not a separate app** in the browser workflow anymore.

Recommended flow:
1. In **Stage 1**, upload `CO` JSON and `PO` JSON.
2. Click **Run Mapping**.
3. Review matrix and pair details.
4. Switch to **Stage 2**, upload CO attainment JSON + config JSON.
5. Click **Run Attainment Analysis** (it reuses Stage 1 matrix by default).
6. Export Stage 1 and Stage 2 result files from the download buttons.

Pairwise mapping threshold scale used by the scorer:
- `0` for `0.00 <= confidence < 0.10`
- `1` for `0.10 <= confidence < 0.30`
- `2` for `0.30 <= confidence < 0.50`
- `3` for `confidence >= 0.50`

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


## Stage 2: Attainment Analysis Engine

This stage consumes:
- CO attainment input (`ma_attainment`, `ea_attainment`, `indirect_attainment`)
- mapping matrix from Stage 1 (`co_id,PO1,PO2,...`)
- configuration weights and target levels

### Formulas

- `DirectCO = (MA * ma_weight) + (EA * ea_weight)`
- `FinalCO = (DirectCO * direct_weight) + (Indirect * indirect_weight)`
- `COScaled = FinalCO * 3`
- `PO = sum(FinalCO_i * Map_ij) / sum(Map_ij)`
- `POScaled = PO * 3`

### Run Stage 2 CLI

```bash
python -m copo_mapper.attainment_cli \
  --co-attainment-file examples/co_attainment.json \
  --mapping-matrix-file examples/mapping_matrix.csv \
  --config-file examples/attainment_config.json \
  --out-dir attainment_outputs
```

Outputs:
- `attainment_outputs/co_attainment_summary.csv`
- `attainment_outputs/po_attainment_summary.csv`
- `attainment_outputs/target_achievement.csv`
- `attainment_outputs/course_summary.json`
