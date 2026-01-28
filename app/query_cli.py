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

context = "\n".join([
        f"[Source: {doc.metadata.get('filename')}] {doc.page_content[:500]}"
        for doc, dist in hits
    ])
    
prompt = f"""Based on the following context from medical records, answer the user's question.
If you need more specific data (like counting ALL patients or getting complete lists), use the search_csv tool.

CONTEXT:
{context}

USER QUESTION: {query}

Answer:"""
    
print("\nCalling agent...")
result = agent.invoke({
    'messages': [HumanMessage(content=prompt)],
    'llm_calls': 0
})
    
print("\n=== AGENT RESPONSE ===")
print(result['messages'][-1].content)


