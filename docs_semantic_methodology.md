# Semantic Backend Methodology (TF-IDF, SBERT, BERT)

This note documents how semantic similarity is computed in the project and how backend selection is enforced.

- TF-IDF: tokenized term-frequency vectors weighted by smooth IDF and compared via cosine similarity.
- SBERT: sentence-transformers embeddings with `normalize_embeddings=True`; cosine is computed as dot product of normalized vectors.
- BERT: transformer hidden states are mean-pooled over attention mask, L2-normalized, then compared via cosine.
- Backend behavior: selecting SBERT/BERT raises an error if model/dependencies cannot load; no fallback to TF-IDF.

## Threshold calibration per backend

The pairwise label is derived from a composite score:

```
composite = 0.45 * semantic_similarity
          + 0.20 * domain_overlap (Jaccard)
          + 0.15 * token_overlap (Jaccard)
          + 0.20 * bloom_proximity (1 - gap/5)
```

so semantic similarity contributes at most 0.45 and the lexical/taxonomic features at most 0.55.

Feature behavior for unrelated texts (measured on the real MBA exports, 1980 CO × PO pairs, tfidf pipeline):

- **Bloom term.** Bloom levels are detected from the *leading verb group* of each statement (mid-sentence noun-homographs like "the value of assets" or "Models of Investment" do not count; see `features.detect_bloom`). With the expanded verb lists most real outcome statements carry a detectable level — real COs measure 43% understand, 21% create, 21% apply, 7% evaluate, 6% analyze, 1% remember — so unrelated pairs usually have a nonzero Bloom gap and the bloom term averages **~0.13** (median 0.12; only 20% of real pairs sit at gap 0). The worst case remains 0.20: a pair whose texts both contain *no* action verb defaults both sides to "understand" (gap 0). An unrelated pair with zero lexical overlap therefore carries a composite of roughly `0.45 * sim + 0.13` typically, bounded above by `0.45 * sim + 0.20`.
- **Domain term.** Catch-all management vocabulary ("management", "managerial", "decision", "strategy", "organizational", ...) was pruned from the domain lexicon, phrases now match through the production path, and a single shared *generic* domain (`features.GENERIC_DOMAINS`) earns half credit. Measured on the real exports the domain term is 0 for 93% of pairs (mean 0.007; above 0.1 for only 1.3% of pairs). Pre-fix, this term pushed 40%+ of a real management-course grid to label 2 at semantic similarity 0; post-fix that grid yields at most one label ≥ 2 at similarity 0 (see `scripts/validate_real_data.py` and `tests/test_real_data_calibration.py`). Program-wide the tfidf pipeline now yields 26% "0", 72% "1", 1.7% "2", 0.2% "3", with every label-3 pair a genuinely strong match.

The original label cutoffs (3 at >= 0.50, 2 at >= 0.30, 1 at >= 0.10) were tuned for TF-IDF cosine, which is near 0 for loosely related texts. Dense-embedding backends occupy very different cosine ranges — SBERT (all-MiniLM-L6-v2) sits around 0.2–0.5 for barely related sentence pairs and 0.5–0.8 for related ones, and mean-pooled BERT-base is higher still due to embedding anisotropy — so reusing the TF-IDF cutoffs systematically inflates labels. `score_pair` therefore takes a `backend` parameter and applies per-backend cutoffs from `scoring.THRESHOLDS` (`{backend: (t3, t2, t1)}`):

| Backend | t3 (label 3) | t2 (label 2) | t1 (label 1) | Sim floor for label 3 |
|---------|--------------|--------------|--------------|-----------------------|
| tfidf   | 0.50         | 0.30         | 0.10         | none (0.0)            |
| sbert   | 0.62         | 0.50         | 0.33         | 0.45                  |
| bert    | 0.72         | 0.60         | 0.48         | 0.70                  |

TF-IDF cutoffs are unchanged, so existing behavior is fully backwards compatible.

### SBERT arithmetic

Based on the known distribution of MiniLM cosines (unrelated < 0.25, weak ~0.25–0.45, moderate ~0.45–0.6, strong > 0.6). Anchors use the *worst-case* bloom term 0.20 (verb-less texts); typical real unrelated pairs carry ~0.13, i.e. sit ~0.07 below these anchors:

- Unrelated: sim 0.20, no overlap → `0.45*0.20 + 0.20 = 0.29` → must map to 0, so t1 > 0.29.
- Weak: sim 0.35, token Jaccard ~0.1 → `0.158 + 0.015 + 0.20 = 0.37` → 1.
- Moderate: sim 0.52, domain Jaccard 0.5, token Jaccard ~0.15 → `0.234 + 0.10 + 0.023 + 0.20 = 0.56` → 2.
- Strong: sim 0.65, domain Jaccard 0.5, token Jaccard ~0.3 → `0.293 + 0.10 + 0.045 + 0.20 = 0.64` → 3.

Cutoffs (0.62, 0.50, 0.33) separate these anchors: `0.29 < 0.33 <= 0.37 < 0.50 <= 0.56 < 0.62 <= 0.64`.

### BERT arithmetic

Mean-pooled `bert-base-uncased` cosines run high even for unrelated text (~0.5–0.6; related pairs ~0.7–0.9). Same worst-case bloom convention as above:

- Unrelated: sim 0.55 → `0.248 + 0.20 = 0.45` → 0.
- Weak: sim 0.65 → `0.293 + 0.20 = 0.49` → 1.
- Moderate: sim 0.75, domain Jaccard 0.5, token Jaccard ~0.15 → `0.66` → 2.
- Strong: sim 0.85, domain Jaccard 0.5, token Jaccard ~0.3 → `0.73` → 3.

Hence cutoffs (0.72, 0.60, 0.48).

### Raw-similarity guardrail for label 3

Because the features alone can contribute up to 0.55, a pair with maximal keyword/Bloom overlap but weak semantics could still cross t3 (e.g. SBERT sim 0.30 gives `0.135 + 0.55 = 0.685 >= 0.62`). To keep label 3 ("strong") backed by genuinely strong semantics, `scoring.SIMILARITY_FLOOR_FOR_3` requires the raw similarity itself to clear a per-backend floor (0.45 for sbert, 0.70 for bert); otherwise the label is capped at 2 and the explanation says so. TF-IDF keeps a floor of 0.0, preserving its historical behavior.

### Empirical validation status

`scripts/calibrate_sbert.py` encodes ten hand-written education-flavored CO/PO pairs of known relatedness (strong/moderate/weak/unrelated), prints the MiniLM similarity distribution per band, and checks the predicted labels against the expected bands. It was not run during calibration because `sentence-transformers` could not be installed in the development sandbox (install exceeded the time budget); the SBERT/BERT cutoffs above rest on the published/well-known cosine distributions of these models rather than a fresh empirical run. Run the script locally once `sentence-transformers` is installed to verify.

The **tfidf** path *is* empirically validated against the real institutional exports: `scripts/validate_real_data.py` runs the full pipeline on real courses (KMBN101 management, KMBFM01 finance, KMBIT04 IT) plus the whole program, prints before/after label distributions, and checks the Bloom and domain acceptance criteria; `tests/test_real_data_calibration.py` locks the same criteria into the test suite (with inline real-text fixtures that run even when the raw exports are absent). The measured feature statistics quoted above come from that harness.

## Common deployment logs (not fatal errors)

When running BERT/SBERT on Streamlit Cloud, you may see:
- `unauthenticated requests to the HF Hub` (means downloads are allowed but rate-limited unless `HF_TOKEN` is set), and
- `UNEXPECTED` keys in a BERT load report (expected when loading a base encoder from a checkpoint that also contains task-specific heads like MLM/NSP).

These messages do not indicate mapping failure by themselves. A true failure is when `run_pairwise_mapping` raises `RuntimeError` because model/dependencies cannot be loaded.
