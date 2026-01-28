from config import Config
from rag.ingest import ingest_folder


cfg = Config()

ingest_folder(
    source_dir=cfg.source_dir,
    file_type=cfg.file_type,
    index_dir=cfg.index_dir,
    embed_model=cfg.embed_model,
    min_chars_per_page=cfg.min_chars_per_page,
    chunk_size=cfg.chunk_size,
    chunk_overlap=cfg.chunk_overlap,
)

print(f"Indexed files from {cfg.source_dir} into {cfg.index_dir}")
