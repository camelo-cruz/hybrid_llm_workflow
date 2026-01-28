from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    source_dir: Path = Path("/Users/alejandra/Documents/GitHub/hybrid_llm_workflow/documents/medical_documents")
    index_dir: Path = Path("data/indexes/medical_documents")
    file_type: str = "csv"
    csv_dir: Path = Path("/Users/alejandra/Documents/GitHub/hybrid_llm_workflow/documents/medical_documents/csv")

    min_chars_per_page: int = 200
    chunk_size: int = 2000
    chunk_overlap: int = 150
    top_k: int = 4

    max_distance: float = 1.5

    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
