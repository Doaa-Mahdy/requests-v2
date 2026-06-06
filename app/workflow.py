from langgraph.graph import StateGraph, END
from app.state import AgentState

from app.nodes.intake import intake_node
from app.nodes.evidence import evidence_node
from app.nodes.vqa import vqa_node
from app.nodes.reasoning import reasoning_node
from app.nodes.report import report_node
from app.agents.search import search_agent

builder = StateGraph(AgentState)

builder.add_node("intake", intake_node)
builder.add_node("evidence", evidence_node)
builder.add_node("vqa", vqa_node)
builder.add_node("search", search_agent)
builder.add_node("reasoning", reasoning_node)
builder.add_node("report", report_node)

builder.set_entry_point("intake")

builder.add_edge("intake", "evidence")
builder.add_edge("evidence", "vqa")
builder.add_edge("vqa", "search")
builder.add_edge("search", "reasoning")
builder.add_edge("reasoning", "report")
builder.add_edge("report", END)

graph = builder.compile()