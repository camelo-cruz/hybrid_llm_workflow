from langgraph.graph import StateGraph, START, END
from agent.agent_flow import llm_call, tool_node, should_continue, MessagesState
from langchain.messages import HumanMessage

agent_builder = StateGraph(MessagesState)

agent_builder.add_node('llm_call', llm_call)
agent_builder.add_node('tool_node', tool_node)

agent_builder.add_edge(START, 'llm_call')
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge('tool_node', 'llm_call')

agent = agent_builder.compile()