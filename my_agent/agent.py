from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from .utils.state import AgentState, create_initial_state
from .utils.nodes import plan_step, execute_step, replan_step, should_continue
from IPython.display import Image, display
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

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
