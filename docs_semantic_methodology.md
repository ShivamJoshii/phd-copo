# Semantic Backend Methodology (TF-IDF, SBERT, BERT)

This note documents how semantic similarity is computed in the project and how backend selection is enforced.

- TF-IDF: tokenized term-frequency vectors weighted by smooth IDF and compared via cosine similarity.
- SBERT: sentence-transformers embeddings with `normalize_embeddings=True`; cosine is computed as dot product of normalized vectors.
- BERT: transformer hidden states are mean-pooled over attention mask, L2-normalized, then compared via cosine.
- Backend behavior: selecting SBERT/BERT raises an error if model/dependencies cannot load; no fallback to TF-IDF.
