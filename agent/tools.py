from rag.retrieve import load_index
from ticketing.memory import InMemoryTicketing
from config import Config
from langchain.tools import tool

cfg = Config()

def make_retrieval_tool(vs, k: int = 4):
    @tool
    def retrieve_sources(query: str) -> str:
        """Retrieve top-k relevant passages from the indexed PDFs. Returns citations."""
        hits = vs.similarity_search_with_score(query, k=k)
        if not hits:
            return "NO_HITS"

        lines = []
        for i, (doc, dist) in enumerate(hits, 1):
            fname = doc.metadata.get("filename")
            page = doc.metadata.get("page")
            snippet = doc.page_content[:400].replace("\n", " ")
            lines.append(f"[{i}] file={fname} page={page} dist={dist:.3f} text={snippet}")
        return "\n".join(lines)

    return retrieve_sources


def make_ticket_tool(ticketing: InMemoryTicketing):
    @tool
    def open_ticket(reason: str, query: str, evidence: str) -> str:
        """Open a ticket when the system cannot answer confidently."""
        t = ticketing.create_ticket(
            type=reason,
            query=query,
            best_distance=float("inf"),
            hits=[],
        )
        return f"TICKET_CREATED id={t.id} type={t.type}"

    return open_ticket