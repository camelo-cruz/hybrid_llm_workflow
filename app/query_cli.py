import sys

from config import Config
from rag.retrieve import load_index, search_with_scores
from rag.decision import decide
from ticketing.memory import InMemoryTicketing
from agent.agent import agent
from wasabi import msg
from langchain.messages import HumanMessage

cfg = Config()
query = " ".join(sys.argv[1:]).strip()
if not query:
    msg.fail("Please provide a query as a command-line argument.", exits=1)
else:
    msg.good(f"Running query: {query}")

if cfg.index_dir is None:
    msg.fail("Index directory is not configured. Please run the ingestion first.", exits=1)

vs = load_index(cfg.index_dir, cfg.embed_model)
hits = search_with_scores(vs, query=query, k=cfg.top_k)

decision = decide(hits, max_distance=cfg.max_distance)

print("\nQUERY:", query)
print("DECISION:", decision.action, "| reason:", decision.reason, "| best_distance:", decision.best_distance)


if decision.action == "ticket":
    ticketing = InMemoryTicketing()
    t = ticketing.create_ticket(
        type="DOC_GAP_OR_LOW_RELEVANCE",
        query=query,
        best_distance=decision.best_distance,
        hits=hits,
    )
    print("\nTICKET CREATED:", t.id)
    print("type:", t.type)
    print("top_sources:", t.top_sources)
    # opcional: salida “user-facing”
    print("\nNo reliable sources found. Ticket created.")
    sys.exit(0)

elif decision.action == "answer":
    context_parts = []
    for i, (doc, dist) in enumerate(hits, start=1):
        fn = doc.metadata.get("filename", "unknown")
        page = doc.metadata.get("page", doc.metadata.get("page_number", ""))
        context_parts.append(
            f"[S{i} | file={fn} | page={page} | dist={dist:.3f}]\n{doc.page_content}"
        )
    context = "\n\n".join(context_parts)

    prompt = f"""You must answer using ONLY the context sources.
    Cite sources after every sentence using the IDs like [S1], [S2] (multiple allowed).
    If the context does not contain the answer, say "I don't know based on the provided documents" and suggest creating a ticket.

    CONTEXT:
    {context}

    QUESTION: {query}

    Return format:
    - Answer: ...
    - Sources: list the used sources (file names and page)
    """

    print("\nCalling agent...")
    result = agent.invoke({
        "messages": [HumanMessage(content=prompt)],
        "llm_calls": 0
    })

    print("\n=== AGENT RESPONSE ===")
    print(result["messages"][-1].content)
    sys.exit(0)

else:
    msg.fail(f"Unknown decision action: {decision.action}", exits=1)