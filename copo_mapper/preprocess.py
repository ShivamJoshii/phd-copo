import re

FILLER_PATTERNS = [
    r"students will be able to",
    r"the learner should",
    r"at the end of the course",
]


def normalize_text(text: str) -> str:
    value = text.lower().strip()
    for pattern in FILLER_PATTERNS:
        value = re.sub(pattern, " ", value)
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value
