from __future__ import annotations

import importlib
import importlib.util
import math
from collections import Counter
from typing import Optional


def _tf(text: str) -> Counter[str]:
    return Counter(text.split())


def _cosine(counter_a: Counter[str], counter_b: Counter[str]) -> float:
    dot = sum(counter_a[t] * counter_b.get(t, 0) for t in counter_a)
    norm_a = math.sqrt(sum(v * v for v in counter_a.values()))
    norm_b = math.sqrt(sum(v * v for v in counter_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def tfidf_pair_similarity(co_texts: list[str], po_texts: list[str]) -> list[float]:
    if len(co_texts) != len(po_texts):
        raise ValueError("co_texts and po_texts must have the same length.")

    # Build IDF from the corpus of all unique texts (smooth IDF: log((N+1)/(df+1))+1)
    corpus = list(set(co_texts) | set(po_texts))
    N = len(corpus)
    df: Counter[str] = Counter()
    for doc in corpus:
        for term in set(doc.split()):
            df[term] += 1
    idf = {term: math.log((N + 1) / (count + 1)) + 1 for term, count in df.items()}
    default_idf = math.log((N + 1) / 1) + 1

    def _tfidf(text: str) -> Counter[str]:
        return Counter({t: cnt * idf.get(t, default_idf) for t, cnt in _tf(text).items()})

    cache: dict[str, Counter[str]] = {}

    def cached_tfidf(text: str) -> Counter[str]:
        if text not in cache:
            cache[text] = _tfidf(text)
        return cache[text]

    return [_cosine(cached_tfidf(co), cached_tfidf(po)) for co, po in zip(co_texts, po_texts, strict=True)]


def sbert_pair_similarity(
    co_texts: list[str],
    po_texts: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Optional[list[float]]:
    """
    Compute cosine similarity using Sentence-BERT embeddings.

    Returns None if sentence-transformers is unavailable in the runtime.
    """
    if len(co_texts) != len(po_texts):
        raise ValueError("co_texts and po_texts must have the same length.")
    if not co_texts:
        return []

    if importlib.util.find_spec("sentence_transformers") is None:
        return None

    sentence_transformers = importlib.import_module("sentence_transformers")
    SentenceTransformer = sentence_transformers.SentenceTransformer
    model = SentenceTransformer(model_name)
    co_embeddings = model.encode(co_texts, convert_to_numpy=True, normalize_embeddings=True)
    po_embeddings = model.encode(po_texts, convert_to_numpy=True, normalize_embeddings=True)

    return [float((co_embeddings[i] * po_embeddings[i]).sum()) for i in range(len(co_texts))]


def bert_pair_similarity(
    co_texts: list[str],
    po_texts: list[str],
    model_name: str = "google-bert/bert-base-uncased",
) -> Optional[list[float]]:
    """
    Compute cosine similarity using BERT encoder outputs with mean pooling.

    Returns None if transformers/torch is unavailable in the runtime.
    """
    if len(co_texts) != len(po_texts):
        raise ValueError("co_texts and po_texts must have the same length.")
    if not co_texts:
        return []

    if importlib.util.find_spec("transformers") is None or importlib.util.find_spec("torch") is None:
        return None

    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    AutoModel = transformers.AutoModel
    AutoTokenizer = transformers.AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    all_texts = co_texts + po_texts
    encoded = tokenizer(all_texts, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**encoded)
        token_embeddings = outputs.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = (token_embeddings * attention_mask).sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1e-9)
        sentence_embeddings = summed / counts
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

    pair_count = len(co_texts)
    co_embeddings = sentence_embeddings[:pair_count]
    po_embeddings = sentence_embeddings[pair_count:]
    cosine_scores = (co_embeddings * po_embeddings).sum(dim=1)
    return [float(value) for value in cosine_scores.tolist()]
