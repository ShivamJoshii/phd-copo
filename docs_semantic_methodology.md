# Semantic Backend Methodology (TF-IDF, SBERT, BERT)

This note documents how semantic similarity is computed in the project and how backend selection is enforced.

- TF-IDF: tokenized term-frequency vectors weighted by smooth IDF and compared via cosine similarity.
- SBERT: sentence-transformers embeddings with `normalize_embeddings=True`; cosine is computed as dot product of normalized vectors.
- BERT: transformer hidden states are mean-pooled over attention mask, L2-normalized, then compared via cosine.
- Backend behavior: selecting SBERT/BERT raises an error if model/dependencies cannot load; no fallback to TF-IDF.

## Common deployment logs (not fatal errors)

When running BERT/SBERT on Streamlit Cloud, you may see:
- `unauthenticated requests to the HF Hub` (means downloads are allowed but rate-limited unless `HF_TOKEN` is set), and
- `UNEXPECTED` keys in a BERT load report (expected when loading a base encoder from a checkpoint that also contains task-specific heads like MLM/NSP).

These messages do not indicate mapping failure by themselves. A true failure is when `run_pairwise_mapping` raises `RuntimeError` because model/dependencies cannot be loaded.
