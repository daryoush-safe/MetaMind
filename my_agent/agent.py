from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from .utils.state import (
    PlanExecutionState,
    AgentPlannerState,
    AgentExecutorState,
)
from .utils.nodes import (
    call_planner,
    should_continue_planner_tools,
    call_executor,
    should_continue_executor_tools,
    planner_step,
    executor_step,
    replan_step,
    should_replan,
)
from .utils.tools.planner.tool_registry import PLANNER_TOOLS
from .utils.tools.executer.tool_registry import EXECUTOR_TOOLS


# =========================================================================
# Sub-graph builders
# =========================================================================

def build_planner_graph():
    builder = StateGraph(AgentPlannerState)

    builder.add_node("planner", call_planner)
    builder.add_node("planner_tools", ToolNode(tools=PLANNER_TOOLS))

    builder.set_entry_point("planner")

    builder.add_conditional_edges(
        "planner",
        should_continue_planner_tools,
        {
            "continue": "planner_tools",
            "end": END,
        },
    )
    builder.add_edge("planner_tools", "planner")

    return builder.compile()


def build_executor_graph():
    builder = StateGraph(AgentExecutorState)

    builder.add_node("executor", call_executor)
    builder.add_node("executor_tools", ToolNode(tools=EXECUTOR_TOOLS))

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
    compiled_planner = build_planner_graph()
    compiled_executor = build_executor_graph()

    def _planner_step(state: PlanExecutionState) -> dict:
        return planner_step(state, compiled_planner)

    def _executor_step(state: PlanExecutionState) -> dict:
        return executor_step(state, compiled_executor)

    graph = StateGraph(PlanExecutionState)

    graph.add_node("planner", _planner_step)
    graph.add_node("agent", _executor_step)
    graph.add_node("replanner", replan_step)

    graph.set_entry_point("planner")

    graph.add_edge("planner", "agent")
    graph.add_edge("agent", "replanner")

    graph.add_conditional_edges(
        "replanner",
        should_replan,
        {
            "continue": "agent",
            "end": END,
        },
    )

    return graph