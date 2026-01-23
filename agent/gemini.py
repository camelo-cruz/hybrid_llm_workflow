import os
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv


load_dotenv('secrets.env')

gemini_key = os.getenv("GEMINI_KEY")

gemini = init_chat_model(
    "google_genai:gemini-2.5-flash-lite",
    temperature=0,
    api_key=gemini_key,
)


@tool
def search_in_file(query: str) -> str:
    """Search for a query in a file and return the results."""
    # Implement the search logic here
    return f"Results for '{query}'"


tools = [search_in_file]

tools_by_name = {tool.name: tool for tool in tools}
gemini_with_tools = gemini.bind_tools(tools)