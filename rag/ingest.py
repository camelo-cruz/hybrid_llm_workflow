# rag/ingest.py
from __future__ import annotations

import re
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def clean_extracted_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    cleaned_lines = []
    for line in text.split("\n"):
        ln = line.strip()
        if not ln:
            continue

        # Drop lines containing only symbols/digits (e.g., page numbers, figure numbers)
        only_symbols_digits = re.sub(r"[0-9\s,.\-:/]", "", ln) == ""
        if only_symbols_digits and len(ln) <= 40:
            continue

        # Drop very short lines that are mostly non-letters
        letters = re.findall(r"[A-Za-zÄÖÜäöüß]", ln)
        if len(letters) < 10 and len(ln) < 25:
            continue

        cleaned_lines.append(ln)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdfs(pdf_dir: Path) -> List[Document]:
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir}")

    docs: List[Document] = []
    for path in pdf_paths:
        pages = PyMuPDFLoader(str(path)).load()
        for d in pages:
            d.page_content = clean_extracted_text(d.page_content)
            d.metadata["source"] = str(path)
            d.metadata["filename"] = path.name
        docs.extend(pages)

    return docs


def filter_pages(docs: List[Document], min_chars_per_page: int) -> List[Document]:
    return [d for d in docs if len(d.page_content.strip()) >= min_chars_per_page]


def chunk_docs(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def build_and_save_index(
    docs: List[Document],
    embed_model: str,
    index_dir: Path,
) -> None:
    embeddings = HuggingFaceEmbeddings(model_name=embed_model)
    vectorstore = FAISS.from_documents(docs, embeddings)

    index_dir.parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))


def ingest_folder(
    pdf_dir: Path,
    index_dir: Path,
    embed_model: str,
    min_chars_per_page: int,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    docs = load_pdfs(pdf_dir)
    kept = filter_pages(docs, min_chars_per_page=min_chars_per_page)
    chunks = chunk_docs(kept, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    build_and_save_index(chunks, embed_model=embed_model, index_dir=index_dir)
