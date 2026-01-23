from langchain.tools import tool
from langchain.chat_models import init_chat_model


qwen = init_chat_model("ollama:qwen2:7b", temperature=0)

# Define tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a / b

tools = [multiply, add, divide]
tools_by_name = {tool.name: tool for tool in tools}
qwen_with_tools = qwen.bind_tools(tools)
