from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from .utils.state import PlanExecutionState, AgentExecutorState
from .utils.tools.tool_registry import ALL_TOOLS
from .utils.nodes import (
    plan_step, execute_step, replan_step, should_continue,
    call_executor, should_continue_executor_tools,
)


# =========================================================================
# Sub-graph builders
# =========================================================================

def build_executor_graph():
    builder = StateGraph(AgentExecutorState)

    builder.add_node("executor", call_executor)
    builder.add_node("executor_tools", ToolNode(tools=ALL_TOOLS))

    builder.set_entry_point("executor")

    builder.add_conditional_edges(
        "executor",
        should_continue_executor_tools,
        {
            "continue": "executor_tools",
            "end": END,
        },
    )
    builder.add_edge("executor_tools", "executor")

    return builder.compile()


# =========================================================================
# Main (outer) graph builder
# =========================================================================

def build_graph() -> StateGraph:
    compiled_executor = build_executor_graph()

    def _execute_step(state: PlanExecutionState) -> dict:
        return execute_step(state, compiled_executor)
    
    graph = StateGraph(PlanExecutionState)

    graph.add_node("planner", plan_step)
    graph.add_node("agent", _execute_step)
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
