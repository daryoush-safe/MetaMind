from langgraph.graph import StateGraph, END
from .utils.state import AgentState
from .utils.nodes import plan_step, execute_step, replan_step, should_continue

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("planner", plan_step)
    graph.add_node("agent", execute_step)
    graph.add_node("replanner", replan_step)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "agent")
    graph.add_edge("agent", "replanner")
    graph.add_conditional_edges(
        "replanner",
        should_continue,
        {
            "continue": "agent",
            "end": END
        }
    )
    return graph
