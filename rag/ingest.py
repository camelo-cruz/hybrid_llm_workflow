# rag/ingest.py
from __future__ import annotations

import re
from pathlib import Path
from typing import List

import pandas as pd

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


def load_files(source_dir: Path, file_type: str = 'pdf', rows_per_chunk: int = 50) -> List[Document]:
    """Load PDF or CSV files from a directory into Documents.
    
    Args:
        source_dir: Directory containing the files to load.
        file_type: Type of files to load ('pdf' or 'csv').
        rows_per_chunk: For CSVs, number of rows to combine into one document.
    
    Returns:
        List of Document objects with content and metadata.
    """
    paths = sorted(source_dir.glob(f"*.{file_type}"))

    if not paths:
        raise FileNotFoundError(f"No {file_type.upper()} files found in {source_dir}")

    docs: List[Document] = []
    for path in paths:
        if file_type == 'pdf':
            pages = PyMuPDFLoader(str(path)).load()
            # Clean extracted text only for PDFs (removes noise like page numbers)
            for d in pages:
                d.page_content = clean_extracted_text(d.page_content)
                d.metadata["source"] = str(path)
                d.metadata["filename"] = path.name
            docs.extend(pages)
            
        elif file_type == 'csv':
            # Use pandas to batch rows together for efficiency
            df = pd.read_csv(path)
            columns = df.columns.tolist()
            
            # Group rows into chunks to reduce document count
            for start_idx in range(0, len(df), rows_per_chunk):
                chunk_df = df.iloc[start_idx:start_idx + rows_per_chunk]
                
                # Format each row as "col1: val1 | col2: val2 | ..."
                rows_text = []
                for _, row in chunk_df.iterrows():
                    row_str = " | ".join(f"{col}: {row[col]}" for col in columns)
                    rows_text.append(row_str)
                
                content = f"File: {path.name}\n" + "\n".join(rows_text)
                
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": str(path),
                        "filename": path.name,
                        "row_start": start_idx,
                        "row_end": min(start_idx + rows_per_chunk, len(df)),
                    }
                )
                docs.append(doc)
            
            pages = len(range(0, len(df), rows_per_chunk))
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        print(f"Loaded {len(pages) if file_type == 'pdf' else pages} documents from {path.name}")

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
    source_dir: Path,
    file_type: str,
    index_dir: Path,
    embed_model: str,
    min_chars_per_page: int,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    print("Loading files...")
    docs = load_files(source_dir, file_type=file_type)
    print(f"Total documents loaded: {len(docs)}")
    
    print("Filtering pages...")
    kept = filter_pages(docs, min_chars_per_page=min_chars_per_page)
    print(f"Documents after filtering: {len(kept)}")
    
    print("Chunking documents...")
    chunks = chunk_docs(kept, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"Total chunks: {len(chunks)}")
    
    print("Building embeddings and index (this may take a few minutes)...")
    build_and_save_index(chunks, embed_model=embed_model, index_dir=index_dir)
    print("Done!")
