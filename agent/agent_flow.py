
import operator
from langchain.messages import AnyMessage, SystemMessage
from typing_extensions import TypedDict, Annotated, NotRequired
from langchain.messages import ToolMessage
from qwen import qwen_with_tools, qwen_tools_by_name
from gemini import gemini_with_tools, gemini_tools_by_name
from langgraph.graph import END


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: Annotated[int, operator.add]

class MessagesUpdate(TypedDict):
    messages: NotRequired[list[AnyMessage]]
    llm_calls: NotRequired[int]

def llm_call(state: MessagesState) -> MessagesUpdate:
    msg = gemini_with_tools.invoke([SystemMessage(content="You are a helpful tasked with looking for text matches inside.")] + state["messages"])
    return {
        "messages": [msg],
        "llm_calls": 1,
    }

def tool_node(state: MessagesState):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = gemini_tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


def should_continue(state: MessagesState):
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tool_node"
    return END