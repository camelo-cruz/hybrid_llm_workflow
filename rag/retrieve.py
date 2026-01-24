from pathlib import Path
from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


def load_index(index_dir: Path, embed_model: str) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=embed_model)
    return FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )

def search_with_scores(vectorstore: FAISS, query: str, k: int) -> List[Tuple[Document, float]]:
    return vectorstore.similarity_search_with_score(query, k=k)
