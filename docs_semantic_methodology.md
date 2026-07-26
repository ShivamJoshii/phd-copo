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

The label cutoffs (3 at >= 0.50, 2 at >= 0.30, 1 at >= 0.10) were tuned for TF-IDF cosine, which is near 0 for loosely related texts, and are empirically validated on the real institutional exports (see below). Dense-embedding backends occupy very different cosine ranges — SBERT (all-MiniLM-L6-v2) sits around 0.2–0.5 for barely related sentence pairs and 0.5–0.8 for related ones, and mean-pooled BERT-base is higher still due to embedding anisotropy — so feeding raw dense cosines into the same composite systematically mislabels pairs.

**Design (current): per-backend similarity normalization, one shared threshold set.** Instead of maintaining separate composite cutoffs per backend, `scoring.rescale_similarity` maps each backend's raw cosine onto a common [0, 1] semantic scale *before* the composite is computed:

```
rescaled = clamp((raw - lo) / (hi - lo), 0, 1)
```

with per-backend anchors in `scoring.SIMILARITY_RESCALE` — `lo` is the backend's "unrelated floor" (raw cosine of clearly unrelated outcome pairs) and `hi` its "near-paraphrase ceiling":

| Backend | lo (unrelated floor) | hi (paraphrase ceiling) | Raw-sim floor for label 3 |
|---------|----------------------|-------------------------|---------------------------|
| tfidf   | 0.0                  | 1.0 (identity — no clamp; legacy path, byte-identical) | none (0.0) |
| sbert   | 0.25                 | 0.75                    | 0.45                      |
| bert    | 0.55                 | 0.90                    | 0.70                      |

Because all backends now feed the same scale, `scoring.THRESHOLDS` contains the *same* validated cutoffs `(0.50, 0.30, 0.10)` for every backend. The dict form is kept only so a backend could be given override cutoffs later; all entries are intentionally identical today. Explanations show both values for non-identity backends, e.g. `semantic=0.80 (raw 0.65)`, so nothing is hidden; the tfidf explanation format is unchanged.

> **Why the previous per-backend thresholds were retired.** The former sbert cutoffs (0.62/0.50/0.33) and bert cutoffs (0.72/0.60/0.48) were derived analytically *assuming the old feature behavior* — a bloom term at its 0.20 worst-case floor and generous domain credit. After the domain-lexicon pruning and Bloom leading-verb fix, the measured feature contribution on the 1,980 real pairs dropped to ~0.13–0.15 (domain term 0 for 93% of pairs, mean 0.007; bloom term mean ~0.13). A genuinely strong sbert pair (raw sim ~0.65, little lexical overlap) then reached only `0.45*0.65 + ~0.15 ≈ 0.44` — below even the old sbert t2 of 0.50 — so nearly every sbert prediction collapsed to 0.

### Band arithmetic on the common scale

Bloom term shown at its measured typical value ~0.13 (worst case 0.20 for verb-less texts).

SBERT (anchors 0.25/0.75):

| Band | Raw sim | Rescaled | Composite | Label |
|------|---------|----------|-----------|-------|
| Unrelated | 0.20 | 0.00 | features only: ~0.13 typical, 0.20 at the bloom-default worst case | 1 (never 2: 0.20 < t2 = 0.30); 0 with any bloom gap >= 3 |
| Weak | 0.35 | 0.20 | `0.09 + ~0.13 = 0.22` | 1 |
| Moderate | 0.50 | 0.50 | `0.225 + ~0.15 = 0.38` | 2 |
| Strong | 0.65 | 0.80 | `0.36 + ~0.18 = 0.54` | 3 (raw 0.65 also clears the 0.45 floor) |

BERT (anchors 0.55/0.90):

| Band | Raw sim | Rescaled | Composite | Label |
|------|---------|----------|-----------|-------|
| Unrelated | 0.55 | 0.00 | <= 0.20 | 0 or low 1 |
| Weak | 0.65 | 0.286 | `0.129 + ~0.13 = 0.26` | 1 |
| Moderate | 0.75 | 0.571 | `0.257 + ~0.15 = 0.41` | 2 |
| Strong | 0.85 | 0.857 | `0.386 + ~0.18 = 0.57` | 3 (raw 0.85 clears the 0.70 floor) |

The bloom-default worst case is explicitly safe: a raw-0.20 sbert pair whose texts both lack an action verb (both default to "understand", bloom term 0.20 exactly) lands at composite 0.20 → label 1, below t2 = 0.30, so verb-less unrelated pairs cannot reach label 2.

### Raw-similarity guardrail for label 3

Because the features alone can contribute up to 0.55, a pair with maximal keyword/Bloom overlap but weak semantics could still cross t3 (e.g. SBERT raw sim 0.30 → rescaled 0.10 gives `0.045 + 0.55 = 0.595 >= 0.50`). To keep label 3 ("strong") backed by genuinely strong semantics, `scoring.SIMILARITY_FLOOR_FOR_3` requires the *raw* similarity itself to clear a per-backend floor (0.45 for sbert — the MiniLM moderate/strong boundary; 0.70 for bert); otherwise the label is capped at 2 and the explanation says so. TF-IDF keeps a floor of 0.0, preserving its historical behavior.

### Empirical validation status

- **tfidf backwards compatibility:** with the identity mapping the tfidf path was verified byte-for-byte against the pre-rescale implementation on the full real program grid (1,980 CO × PO pairs from the institutional exports): `pair_predictions.csv` and `matrix.csv` are identical files.
- **sbert/bert anchors:** an empirical run was again not possible in the calibration sandbox during this revision — the sandbox network allowlist blocks both the PyTorch CPU wheel index and `huggingface.co` (the model-weight host), so neither `sentence-transformers` nor the MiniLM checkpoint could be fetched. The (0.25, 0.75) MiniLM and (0.55, 0.90) mean-pooled-BERT anchors therefore rest on the published/well-known cosine distributions of these model families (MiniLM: unrelated < ~0.25, near-paraphrase > ~0.75; mean-pooled bert-base: unrelated rarely below ~0.55, near-paraphrase ~0.90), sanity-checked through the band arithmetic above. `scripts/calibrate_sbert.py` re-derives them empirically: it encodes ten hand-written education-flavored CO/PO pairs of known relatedness (strong/moderate/weak/unrelated), prints the raw-vs-rescaled similarity distribution per band, and checks predicted labels against the expected bands. Run it once `sentence-transformers` is installed; if the observed unrelated-band maximum or strong-band minimum drifts materially from the anchors, adjust `scoring.SIMILARITY_RESCALE` from the observed percentiles.

The **tfidf** path *is* empirically validated against the real institutional exports: `scripts/validate_real_data.py` runs the full pipeline on real courses (KMBN101 management, KMBFM01 finance, KMBIT04 IT) plus the whole program, prints before/after label distributions, and checks the Bloom and domain acceptance criteria; `tests/test_real_data_calibration.py` locks the same criteria into the test suite (with inline real-text fixtures that run even when the raw exports are absent). The measured feature statistics quoted above come from that harness.

## Common deployment logs (not fatal errors)

When running BERT/SBERT on Streamlit Cloud, you may see:
- `unauthenticated requests to the HF Hub` (means downloads are allowed but rate-limited unless `HF_TOKEN` is set), and
- `UNEXPECTED` keys in a BERT load report (expected when loading a base encoder from a checkpoint that also contains task-specific heads like MLM/NSP).

These messages do not indicate mapping failure by themselves. A true failure is when `run_pairwise_mapping` raises `RuntimeError` because model/dependencies cannot be loaded.
