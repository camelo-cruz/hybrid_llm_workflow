import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from ticketing.memory import InMemoryTicketing
from rag.retrieve import load_index
from config import Config
from agent.tools import make_retrieval_tool, make_ticket_tool


cfg = Config()


load_dotenv("secrets.env")

if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in secrets.env")

gemini = init_chat_model(
    "google_genai:gemini-2.5-flash-lite",
    temperature=0
)

vs = load_index(cfg.index_dir, cfg.embed_model)
ticketing = InMemoryTicketing()

retrieve_tool = make_retrieval_tool(vs, k=cfg.top_k)
ticket_tool = make_ticket_tool(ticketing)

tools = [retrieve_tool, ticket_tool]
gemini_tools_by_name = {t.name: t for t in tools}

gemini_with_tools = gemini.bind_tools(tools)

q = "How can I save energy?"
print(retrieve_tool.invoke({"query": q}))

