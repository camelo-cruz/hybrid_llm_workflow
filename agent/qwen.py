from langchain_ollama import ChatOllama
from config import Config
from rag.retrieve import load_index
from ticketing.memory import InMemoryTicketing
from agent.tools import make_retrieval_tool, make_ticket_tool

cfg = Config()

qwen = ChatOllama(model="qwen2:7b", temperature=0)

vs = load_index(cfg.index_dir, cfg.embed_model)
ticketing = InMemoryTicketing()

retrieve_tool = make_retrieval_tool(vs, k=cfg.top_k)
ticket_tool = make_ticket_tool(ticketing)

tools = [retrieve_tool, ticket_tool]
qwen_tools_by_name = {tool.name: tool for tool in tools}
qwen_with_tools = qwen.bind_tools(tools)

q = "How can I save energy?"
print(retrieve_tool.invoke({"query": q}))
