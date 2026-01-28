from config import Config
from rag.ingest import ingest_pdf_folder


cfg = Config()

ingest_pdf_folder(
    pdf_dir=cfg.pdf_dir,
    index_dir=cfg.index_dir,
    embed_model=cfg.embed_model,
    min_chars_per_page=cfg.min_chars_per_page,
    chunk_size=cfg.chunk_size,
    chunk_overlap=cfg.chunk_overlap,
)

print(f"Indexed PDFs from {cfg.pdf_dir} into {cfg.index_dir}")
