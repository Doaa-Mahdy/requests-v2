
from langgraph.graph import StateGraph, END
from app.state import CaseState

# Import your nodes
from app.nodes.intake import intake_node       # Our new deterministic multi-modal tool runner
from app.nodes.evidence import evidence_node
from app.nodes.vqa import vqa_node
from app.agents.search import search_agent        # Your autonomous search sub-graph agent
from app.nodes.reasoning import reasoning_node
from app.nodes.report import report_node


def decide_next_step(state: CaseState):
    step = state.get("reasoning", {}).get("next_step")
    return step if step in {"vqa", "search", "report"} else "report"

builder = StateGraph(CaseState)

# Add processing nodes to the graph blueprint
builder.add_node("intake", intake_node)
builder.add_node("evidence", evidence_node)
builder.add_node("vqa", vqa_node)
builder.add_node("search", search_agent)
builder.add_node("reasoning", reasoning_node)
builder.add_node("report", report_node)

# Configure the active inquiry loop execution flow
builder.set_entry_point("intake") # Intake gathers raw materials first

builder.add_edge("intake", "evidence")
builder.add_edge("evidence", "reasoning")

builder.add_conditional_edges(
    "reasoning",
    lambda state: state["reasoning"]["next_step"],
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