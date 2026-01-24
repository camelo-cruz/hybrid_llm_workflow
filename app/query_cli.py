import sys

from config import Config
from rag.retrieve import load_index, search_with_scores
from rag.decision import decide
from ticketing.memory import InMemoryTicketing
from agent.agent import agent


cfg = Config()
query = " ".join(sys.argv[1:]).strip() or "I like chocolate"

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
else:

    # Call the agent to answer the query
    # TODO: pass hits to the agent to use as context

    print("\nTOP HITS:")
    for i, (doc, dist) in enumerate(hits, 1):
        print(f"\n--- HIT {i} ---")
        print("dist:", dist)
        print("file:", doc.metadata.get("filename"))
        print("page:", doc.metadata.get("page"))
        print("preview:", doc.page_content[:300])

    print("We need to call the agent to answer the query here...")


