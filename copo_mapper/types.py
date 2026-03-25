from dataclasses import dataclass


@dataclass(frozen=True)
class Outcome:
    id: str
    text: str


@dataclass(frozen=True)
class PairRecord:
    co_id: str
    co_text: str
    po_id: str
    po_text: str
