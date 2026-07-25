# CO-PO Pairwise Mapping System

This repository contains a practical baseline implementation of the **CO-PO pairwise mapping framework** for Outcome-Based Education (OBE).

## What it does

Given a list of Course Outcomes (COs) and Program Outcomes (POs), the system:

1. Builds the full Cartesian product of CO-PO pairs.
2. Extracts educational features (action intent, Bloom level, domain overlap).
3. Computes semantic similarity (TF-IDF cosine).
4. Predicts mapping strength on a 4-point scale (`0,1,2,3`).
5. Exports pairwise predictions and a matrix view.
6. Supports optional SBERT or BERT semantic similarity (strict backend execution for fair comparison).

## Project status

This is an MVP with:

- a default transparent baseline (TF-IDF-style cosine over normalized text tokens),
- optional SBERT or BERT semantic embeddings,
- rule-based educational feature fusion for final 4-class mapping.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

To enable SBERT/BERT backends:

```bash
pip install -e ".[nlp]"
```

### Run pairwise prediction (rule-based baseline)

```bash
copo-map \
  --co-file examples/co.json \
  --po-file examples/po.json \
  --out-dir outputs
```

### Run with SBERT semantic similarity

```bash
copo-map \
  --co-file examples/co.json \
  --po-file examples/po.json \
  --out-dir outputs \
  --semantic-backend sbert \
  --semantic-model sentence-transformers/all-MiniLM-L6-v2
```

### Run with BERT semantic similarity

```bash
copo-map \
  --co-file examples/co.json \
  --po-file examples/po.json \
  --out-dir outputs \
  --semantic-backend bert \
  --semantic-model google-bert/bert-base-uncased
```

If selected neural dependencies are unavailable, the pipeline raises an error for SBERT/BERT so method comparisons remain valid.

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



### Streamlit deployment (auto-install dependencies)

If you deploy on Streamlit Cloud, include a `requirements.txt` in repo root.
This repo now includes one with `streamlit`, `sentence-transformers`, `transformers`, and `torch`, so the deploy environment installs BERT/SBERT dependencies automatically.

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

## Methodology (what the system is using and how it works)

For each CO-PO pair, the scoring pipeline works as follows:

1. **Text preprocessing**  
   CO and PO descriptions are normalized (case-folding, punctuation cleanup, token normalization) before feature extraction.
2. **Pair generation**  
   The full Cartesian product of COs and POs is built, so every CO is compared against every PO.
3. **Semantic similarity (Layer 3)**  
   - **Default**: TF-IDF-style cosine proxy over token-frequency vectors.  
   - **Optional (`--semantic-backend sbert`)**: SBERT sentence embeddings using a Sentence-Transformers model (default: `all-MiniLM-L6-v2`) and cosine in embedding space.  
   - **Optional (`--semantic-backend bert`)**: BERT encoder hidden states (`bert-base-uncased` by default), mean-pooled over valid tokens, then cosine in embedding space.
4. **Educational feature extraction (Layer 4/5)**  
   - Bloom/action-intent detection per outcome and Bloom gap between CO and PO,  
   - domain keyword overlap (Jaccard),  
   - lexical token overlap (Jaccard).
5. **Composite confidence and label mapping (Layer 6)**  
   A weighted composite score is computed:  
   `0.45*semantic + 0.20*domain + 0.15*token + 0.20*bloom_alignment`  
   Then mapped to strengths: `0/1/2/3` using fixed thresholds.
6. **Outputs**  
   - `pair_predictions.csv` with per-pair score, confidence, explanation, and semantic method used,  
   - `matrix.csv` as CO × PO strength table.

### SBERT vs BERT: technical and procedural differences

- **Training objective / representation quality**  
  - **SBERT**: fine-tuned specifically for sentence-level similarity/retrieval, so sentence embeddings are directly optimized for cosine comparison.  
  - **BERT**: base encoder pretraining is token-level (MLM/NSP style), so sentence embeddings are constructed indirectly via pooling.
- **Embedding construction in this repo**  
  - **SBERT path**: uses `SentenceTransformer.encode(..., normalize_embeddings=True)` then dot product (= cosine on normalized vectors).  
  - **BERT path**: tokenizes text, runs `AutoModel`, mean-pools last hidden states with attention mask, L2-normalizes, then cosine.
- **Typical trade-offs**  
  - **SBERT** generally gives stronger semantic similarity quality out of the box for pair-matching tasks.  
  - **BERT** gives flexibility to use any encoder checkpoint but may require more tuning or task-specific fine-tuning for best similarity accuracy.

## Architecture mapping to specification

- **Layer 1**: preprocessing (`copo_mapper/preprocess.py`)
- **Layer 2**: structural extraction (`copo_mapper/features.py`)
- **Layer 3**: semantic representation (`copo_mapper/semantic.py`)
- **Layer 4/5**: pair scoring and educational features (`copo_mapper/scoring.py`)
- **Layer 6**: final 4-class decision (`copo_mapper/scoring.py`)

## Next milestones

1. Add cross-encoder pair scorer.
2. Add trainable XGBoost classifier on labeled faculty data.
3. Build review UI/API for human corrections and feedback loop.




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

Stage 1 sidebar also includes:
- **Semantic Backend** selector (`tfidf`, `sbert`, `bert`)
- **Semantic Model** optional override input (uses backend default when left empty)
- Upload parser supports common text encodings (`UTF-8`, `UTF-8 BOM`, `CP1252`, `Latin-1`) for CSV/JSON files

Recommended flow:
1. In **Stage 1**, upload `CO` and `PO` files (JSON or CSV).
2. Click **Run Mapping**.
3. Review matrix and pair details.
4. Switch to **Stage 2**. The CO list is seeded from the Stage 1 matrix.
5. Type MA / EA / Indirect values per CO directly in the table (or use
   *Pre-fill CO Attainment* to load a CSV/JSON into the table; you can still edit).
6. Set the `MA weight`, `Direct weight`, and `Target level` inline. `EA weight` and
   `Indirect weight` are derived as `1 - MA weight` and `1 - Direct weight`.
7. Click **Run Attainment Analysis**.
8. Review the CO summary, PO summary (includes an `Attainment in Percentage` column),
   target achievement, and course summary, then download the CSVs.

Program Specific Outcomes (PSOs) are modeled as extra rows in the **Stage 1 PO**
input file — e.g. rows with `PO = PSO1, PSO2, ...`. They become additional
columns in the mapping matrix and are attained just like POs in Stage 2.

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

- `DirectCO = ((MA * ma_weight) + (EA * ea_weight)) / (ma_weight + ea_weight)`
- `FinalCO = (DirectCO * direct_weight) + (Indirect * indirect_weight)`
- `COScaled = FinalCO * 3`
- `PO = sum(FinalCO_i * Map_ij) / sum(Map_ij)`
- `POScaled = PO * 3`

The `DirectCO` formula divides by the sum of internal + external weights so the result stays on the same scale as the inputs regardless of how the institution splits the two (e.g. 30/50 or 40/60). Existing configs whose `ma_weight + ea_weight = 1.0` behave identically to before.

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


## Stage 3: Semester-level PO Attainment

Rolls per-course PO values into a semester score, credit-weighted across the courses in that semester.

### Formula

```
PO_sem = sum(PO_course * credits) / sum(credits)
```

### Input

A single CSV/JSON, one row per course:

```csv
course_id,credits,PO1,PO2,PO3
DBMS,4,2.63,2.63,2.72
OS,3,2.50,2.40,2.60
```

Missing cells are skipped (treated as "course doesn't contribute to that PO"), not as zero.

### Run

```bash
python -m copo_mapper.semester_cli \
  --courses-file examples/courses_semester1.csv \
  --out-dir semester_outputs
```

Output: `semester_outputs/semester_po_attainment.csv` with columns `level, po_id, value, percentage, scaled`.


## Stage 4: Program-level PO Attainment

Rolls semester PO values into a program score, credit-weighted across semesters.

### Formula

```
PO_program = sum(PO_sem * credits) / sum(credits)
```

### Input

```csv
semester_id,credits,PO1
Sem1,20,2.50
Sem2,22,2.58
```

### Run

```bash
python -m copo_mapper.program_cli \
  --semesters-file examples/semesters_program.csv \
  --out-dir program_outputs
```

Output: `program_outputs/program_po_attainment.csv`.


## Root-cause diagnosis & systemic drivers (why a target was missed)

Because every attainment number is a deterministic weighted average, the reason a
CO or PO missed its target is *exactly computable* — no model guessing required.

### Per-course diagnosis (`copo_mapper/diagnostics.py`)

After running Stage 2, the Streamlit app shows a **"Why did targets miss?"** panel:

- **Missed CO** → the weakest input (MA / EA / Indirect) and the exact value a
  single input would need to reach the target (cheapest fix first).
- **Missed PO** → each contributing CO's *weight share* and *drag* (low CO
  attainment × high mapping strength = the real culprit), plus the cheapest
  single CO lever to cross target.

Every lever is verified by construction: applying the suggested value reaches the
target in the same arithmetic the pipeline uses (see `tests/test_diagnostics.py`).

### Systemic drivers (`copo_mapper/ml_drivers.py`)

The interpretable, small-data ML layer. As you analyse multiple courses in the
app, it accumulates CO observations and ranks which input most *systematically*
tracks with missed targets across courses (point-biserial correlation + mean
gap between met/missed groups). It is dependency-free and honest about small
samples; an optional `fit_logistic` upgrade path uses scikit-learn when present
and more data is available. This is the seam for a future trained model.

## End-to-end flow

The whole pipeline forms a funnel:

```
Internal + External    →  Direct
Direct + Indirect      →  Final CO
CO + Mapping           →  Course PO     (Stage 2)
Course PO + Credits    →  Semester PO   (Stage 3)
Semester PO + Credits  →  Program PO    (Stage 4)
```

Every stage is a weighted average — only the weights change (internal/external split, then direct/indirect, then mapping strength, then credits, then credits again).

In the Streamlit app the four stages appear as Steps 1 → 4. Each step's outputs are pushed forward into the next via session state, so you can walk the full flow without re-uploading.
