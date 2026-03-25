"""
LangGraph-based customer support agent.
This agent exists to generate realistic traces — it's the data source, not the product.
"""

from __future__ import annotations

import os
from operator import add
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from agentwatch.agent.tools import AGENT_TOOLS


class AgentState(TypedDict):
    messages: Annotated[list, add]


SYSTEM_PROMPT = """You are a customer support agent for a SaaS platform.
You have access to a knowledge base, customer lookup, system status checks, and ticket creation.

Guidelines:
- Search the knowledge base before answering technical questions
- Look up the customer when they mention account-specific issues
- Check system status if the issue might be infrastructure-related
- Create a ticket for complex issues that need escalation
- Be helpful, specific, and reference the information you found"""


def create_support_agent(model_name: str = "gpt-4o-mini") -> StateGraph:
    """Build the LangGraph support agent."""
    llm = ChatOpenAI(
        model=model_name,
        temperature=0.3,
        api_key=os.getenv("OPENAI_API_KEY"),
    ).bind_tools(AGENT_TOOLS)

    tool_node = ToolNode(AGENT_TOOLS)

    def agent_node(state: AgentState) -> dict:
        msgs = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        resp = llm.invoke(msgs)
        return {"messages": [resp]}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()

