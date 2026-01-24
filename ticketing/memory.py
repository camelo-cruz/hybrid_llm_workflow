# ticketing/memory.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
import time
import uuid

from langchain_core.documents import Document


@dataclass
class Ticket:
    id: str
    type: str
    query: str
    best_distance: float
    top_sources: List[Dict[str, Any]]
    created_at: float = field(default_factory=time.time)


class InMemoryTicketing:
    def __init__(self) -> None:
        self.tickets: List[Ticket] = []

    def create_ticket(
        self,
        type: str,
        query: str,
        best_distance: float,
        hits: List[Tuple[Document, float]],
    ) -> Ticket:
        top_sources: List[Dict[str, Any]] = []
        for doc, dist in hits[:3]:
            top_sources.append(
                {
                    "filename": doc.metadata.get("filename"),
                    "page": doc.metadata.get("page"),
                    "distance": dist,
                    "preview": doc.page_content[:200],
                }
            )

        t = Ticket(
            id=f"T-{uuid.uuid4().hex[:8]}",
            type=type,
            query=query,
            best_distance=best_distance,
            top_sources=top_sources,
        )
        self.tickets.append(t)
        return t
    
    def list_tickets(self) -> List[Ticket]:
        return self.tickets
    
    def empty_tickets(self) -> None:
        self.tickets.clear()