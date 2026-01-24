from dataclasses import dataclass
from typing import List, Tuple

from langchain_core.documents import Document


@dataclass(frozen=True)
class Decision():
    action: str
    reason: str
    best_distance: float


def decide(hits: List[Tuple[Document, float]], max_distance: float) -> Decision:
    if not hits:
        return Decision(action="ticket", reason="no_hits", best_distance=float("inf"))

    best_dist = hits[0][1]
    if best_dist > max_distance:
        return Decision(action="ticket", reason="low_relevance", best_distance=best_dist)

    return Decision(action="answer", reason="sufficient_evidence", best_distance=best_dist)
