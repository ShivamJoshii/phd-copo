# Repository Setup Overview

This document explains the existing setup of the **CO-PO Pairwise Mapping System** repository: what's in it, how the pieces fit together, and how to run / develop it.

## 1. What this repo is

A baseline Python implementation of a **CO-PO (Course Outcome – Program Outcome) pairwise mapping framework** for Outcome-Based Education (OBE). It does two things:

- **Stage 1 – Mapping**: given Course Outcomes (COs) and Program Outcomes (POs), score every CO-PO pair on a 0/1/2/3 scale using semantic similarity + educational features (Bloom level, domain overlap, lexical overlap).
- **Stage 2 – Attainment**: given CO attainment values and the Stage 1 mapping matrix, compute CO/PO attainment, scaled scores, and target-achievement reports.

Both stages are available as CLIs and inside one Streamlit app.

## 2. Top-level layout

```
phd-copo/
├── README.md                       # User-facing docs (usage, formulas, methodology)
├── docs_semantic_methodology.md    # Notes on TF-IDF / SBERT / BERT backends
├── pyproject.toml                  # Package metadata, CLI entry points, extras
├── requirements.txt                # Streamlit Cloud deps (streamlit + nlp stack)
├── streamlit_app.py                # Streamlit UI (Stage 1 + Stage 2 tabs)
├── copo_mapper/                    # Library package
├── examples/                       # Sample CO/PO/attainment inputs (JSON + CSV)
├── tests/                          # unittest suite (smoke + unit tests)
└── .devcontainer/devcontainer.json # Codespaces / dev container config
```

## 3. The `copo_mapper` package

Layered to match the spec described in the README:

| File | Layer | Purpose |
|------|-------|---------|
| `preprocess.py` | L1 | `normalize_text` — case-fold, strip filler phrases ("students will be able to"...), collapse punctuation/whitespace. |
| `features.py` | L2 | Bloom-verb detection, domain keyword detection, token sets, Jaccard, Bloom distance. |
| `semantic.py` | L3 | Three pluggable similarity backends: `tfidf_pair_similarity` (smooth-IDF cosine), `sbert_pair_similarity` (sentence-transformers, normalized embeddings → dot), `bert_pair_similarity` (HF AutoModel, mean-pool, L2-norm, cosine). SBERT/BERT use lazy `importlib` lookups and return `None` if deps are missing. |
| `scoring.py` | L4/5/6 | `score_pair`: composite = `0.45*semantic + 0.20*domain + 0.15*token + 0.20*(1 - bloom_gap/5)`, then thresholded to 0/1/2/3 (cut-offs 0.10 / 0.30 / 0.50). |
| `pipeline.py` | — | `run_pairwise_mapping`: loads CO/PO inputs (JSON or CSV), builds the Cartesian product, picks a semantic backend (raises `RuntimeError` for SBERT/BERT if model/deps can't load — no silent fallback), scores every pair, writes `pair_predictions.csv` and `matrix.csv`. |
| `attainment.py` | Stage 2 | Dataclasses + functions for the attainment math: `DirectCO = (MA*w_ma + EA*w_ea) / (w_ma + w_ea)` (normalized so weights can sum to anything), `FinalCO = Direct*w_d + Indirect*w_i`, `PO = Σ(FinalCO_i * Map_ij) / Σ Map_ij`, scaling × 3, and target-achieved Y/N. Writes 4 outputs: CO summary, PO summary, target achievement, course summary JSON. |
| `aggregate.py` | Stage 3 + 4 | Scale-agnostic credit-weighted aggregation. `CreditRow` dataclass, `aggregate_by_credits`, `compute_semester_po`, `compute_program_po`, `load_credit_rows` (auto-detects id/credits columns from CSV/JSON), `run_semester_aggregation`, `run_program_aggregation`. |
| `cli.py` | — | `copo-map` entry point: `--co-file`, `--po-file`, `--out-dir`, `--semantic-backend {tfidf,sbert,bert}`, `--semantic-model`. |
| `attainment_cli.py` | — | `copo-attainment` entry point: `--co-attainment-file`, `--mapping-matrix-file`, `--config-file`, `--out-dir`. |
| `semester_cli.py` | — | `copo-semester` entry point: `--courses-file`, `--out-dir`. |
| `program_cli.py` | — | `copo-program` entry point: `--semesters-file`, `--out-dir`. |
| `io_utils.py` | — | `normalize_keys` — case-insensitive row keys. |
| `types.py` | — | `Outcome`, `PairRecord` dataclasses. |

Backend selection rule (important): TF-IDF is computed unconditionally as the baseline. If `sbert` or `bert` is requested, that backend's similarities **replace** the TF-IDF ones; if the model/deps can't load, the pipeline raises rather than falling back — this keeps method comparisons fair.

## 4. Configuration files

### `pyproject.toml`
- Package name `copo-mapper`, Python `>=3.10`.
- Core deps: **none** (the baseline is pure stdlib).
- Optional extras:
  - `ui` → `streamlit>=1.30`
  - `nlp` → `sentence-transformers`, `transformers`, `torch`
- Console scripts:
  - `copo-map` → `copo_mapper.cli:main`
  - `copo-attainment` → `copo_mapper.attainment_cli:main`
  - `copo-semester` → `copo_mapper.semester_cli:main`
  - `copo-program` → `copo_mapper.program_cli:main`

### `requirements.txt`
Used by Streamlit Cloud to auto-install the full stack (streamlit + the nlp extras) on deploy.

### `.devcontainer/devcontainer.json`
- Base image: `mcr.microsoft.com/devcontainers/python:1-3.11-bookworm`.
- On content update: installs `packages.txt` (if present) and `requirements.txt`, plus `streamlit`.
- On attach: auto-runs `streamlit run streamlit_app.py` on port `8501` and opens a preview.
- Recommended VS Code extensions: `ms-python.python`, `ms-python.vscode-pylance`.

### `.gitignore`
Ignores `__pycache__/`, `*.pyc`, `outputs/`, `.venv/`, `attainment_outputs/`.

## 5. Input / output formats

Inputs accept **JSON or CSV**; parser is picked by file extension.

- **Stage 1**: `co.{json,csv}` with `CO,description`; `po.{json,csv}` with `PO,description`. (PSOs go in as extra rows like `PSO1, PSO2, ...`.)
- **Stage 2**: `co_attainment.{json,csv}` with `co_id, ma_attainment, ea_attainment, indirect_attainment`; `attainment_config.{json,csv}` with the six weights/targets; `mapping_matrix.csv` produced by Stage 1.

Stage 1 outputs `outputs/pair_predictions.csv` + `outputs/matrix.csv`. Stage 2 outputs `attainment_outputs/co_attainment_summary.csv`, `po_attainment_summary.csv`, `target_achievement.csv`, `course_summary.json`.

The `examples/` directory ships small fixtures (`co.json`, `po.json`) and a larger `co_large` / `po_large` pair for dynamic-size testing (4×5 = 20 pair rows).

## 6. Streamlit app

`streamlit_app.py` is a single page with a left-to-right **4-step flow**:
1. **Step 1: Mapping** — uploads CO/PO, picks backend (`tfidf` / `sbert` / `bert`) + optional model override in the sidebar, runs the pipeline, shows the matrix and per-pair details.
2. **Step 2: Course Attainment** — seeds CO list from the Step 1 matrix automatically (or accepts an uploaded matrix), exposes editable MA/EA/Indirect values and weight/target inline, then runs the attainment analysis. `EA weight` and `Indirect weight` derive as `1 - x`. After running, a *Push to Step 3* control adds the course's per-PO values (with credits) into the semester roll-up.
3. **Step 3: Semester** — editable table of `{course_id, credits, PO1, PO2, ...}`, pre-filled from Step 2 pushes (or uploaded CSV). Runs `compute_semester_po`. A *Push to Step 4* control forwards the semester's per-PO values into the program roll-up.
4. **Step 4: Program** — editable table of `{semester_id, credits, PO1, PO2, ...}`, pre-filled from Step 3 pushes (or uploaded CSV). Runs `compute_program_po`.

## 7. Tests

`tests/` runs under stdlib `unittest`:
- `test_smoke.py` — end-to-end pipeline using temp files.
- `test_dynamic_sizes.py` — verifies N×M pair count for arbitrary sizes.
- `test_semantic.py` — semantic-backend behavior.
- `test_attainment.py` — Stage 2 math.
- `test_aggregate.py` — Direct-formula normalization + Stage 3/4 aggregation, with assertions pinned to the worked-example numbers (CO1 Direct=2.6875, PO1 course=2.633, PO1 semester=2.5743, PO1 program=2.5419).

Run with:
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## 8. Getting started locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .                # baseline (stdlib only)
pip install -e ".[nlp]"         # add SBERT/BERT support
pip install -e ".[ui]"          # add Streamlit

# Stage 1
copo-map --co-file examples/co.json --po-file examples/po.json --out-dir outputs

# Stage 2
copo-attainment \
  --co-attainment-file examples/co_attainment.json \
  --mapping-matrix-file examples/mapping_matrix.csv \
  --config-file examples/attainment_config.json \
  --out-dir attainment_outputs

# Stage 3 (semester)
copo-semester --courses-file examples/courses_semester1.csv --out-dir semester_outputs

# Stage 4 (program)
copo-program --semesters-file examples/semesters_program.csv --out-dir program_outputs

# UI
streamlit run streamlit_app.py
```

## 9. Deployment notes

On Streamlit Cloud the `requirements.txt` triggers automatic install of `streamlit`, `sentence-transformers`, `transformers`, and `torch`, so SBERT/BERT backends work out of the box. Per `docs_semantic_methodology.md`, two log messages there are expected (not errors): unauthenticated HF Hub downloads (rate-limited unless `HF_TOKEN` set) and `UNEXPECTED` keys when loading a base encoder from a checkpoint that contains task heads. A real failure is `run_pairwise_mapping` raising `RuntimeError` from a backend that couldn't load.
