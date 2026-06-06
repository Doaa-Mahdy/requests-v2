from langgraph.graph import StateGraph, END
from app.state import CaseState

from app.nodes.intake import intake_node
from app.nodes.evidence import evidence_node
from app.nodes.vqa import vqa_node
from app.nodes.reasoning import reasoning_node
from app.nodes.report import report_node
from app.agents.search import search_agent


def decide_next_step(state: CaseState):
    reasoning = state.get("reasoning", {})
    loop_count = reasoning.get("loop_count", 0)
    
    if loop_count >= 6 or reasoning.get("next_step") == "report":
        return "report"
    
    return reasoning.get("next_step", "report")

builder = StateGraph(CaseState)

builder.add_node("intake", intake_node)
builder.add_node("evidence", evidence_node)
builder.add_node("vqa", vqa_node)
builder.add_node("search", search_agent)
builder.add_node("reasoning", reasoning_node)
builder.add_node("report", report_node)

builder.set_entry_point("intake")

builder.add_edge("intake", "evidence")
builder.add_edge("evidence", "reasoning")

builder.add_conditional_edges(
    "reasoning",
    decide_next_step,
    {
        "vqa": "vqa",
        "search": "search",
        "report": "report"
    }
)

builder.add_edge("vqa", "reasoning")
builder.add_edge("search", "reasoning")
builder.add_edge("report", END)

graph = builder.compile()