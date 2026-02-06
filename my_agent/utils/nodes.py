import os
import json
import numpy as np
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage
)
from langgraph.prebuilt import ToolNode

from .state import (
    PlanExecutionState,
    AgentPlannerState,
    AgentExecutorState,
    Plan,
    PlanStep,
    ExecutionResult,
)
from .tools.planner.tool_registry import PLANNER_TOOLS
from .tools.executer.tool_registry import EXECUTOR_TOOLS
from .prompts import (
    executor_sys_prompt,
    planner_sys_prompt,
    replanner_sys_prompt,
)


# ============================================================================
# 1.  Utility helpers
# ============================================================================

def convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [convert_numpy_types(i) for i in obj]
        return type(obj)(converted) if isinstance(obj, tuple) else converted
    return obj


def _extract_json(text: str) -> dict:
    """Best-effort JSON extraction from LLM output."""
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    return json.loads(text.strip())


# ============================================================================
# 2.  Lazy LLM / model helpers
#     Called at graph-build time (after dotenv is loaded), not at import time.
# ============================================================================

def _get_llm(temperature: float = 0, model_env: str = "MODEL_NAME") -> ChatOpenAI:
    """Create a ChatOpenAI instance.  Safe to call after load_dotenv()."""
    return ChatOpenAI(
        model=os.environ.get(model_env, os.environ.get("MODEL_NAME")),
        openai_api_key=os.environ.get("API_KEY"),
        openai_api_base=os.environ.get("API_BASE_URL"),
        temperature=temperature,
    )


def get_planner_model():
    """Return planner LLM bound to planner tools."""
    return _get_llm(temperature=0, model_env="PLANNER_MODEL").bind_tools(PLANNER_TOOLS)


def get_executor_model():
    """Return executor LLM bound to executor tools."""
    return _get_llm(temperature=0, model_env="EXECUTOR_MODEL").bind_tools(EXECUTOR_TOOLS)


def get_replanner_model():
    """Return replanner LLM bound to planner tools (for data re-reading)."""
    return _get_llm(temperature=0, model_env="MODEL_NAME").bind_tools(PLANNER_TOOLS)


# ============================================================================
# 3.  Planner sub-graph nodes  (mini ReAct)
# ============================================================================

def call_planner(state: AgentPlannerState) -> dict:
    """Invoke the planner LLM.  It calls data tools then submit_plan."""
    planner_model = get_planner_model()
    messages = list(state["messages"])

    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [
            SystemMessage(content=planner_sys_prompt.PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Analyse this problem and create an execution plan:\n\n"
                f"{state['user_input']}\n\n"
                f"If external data is referenced (.tsp / .csv file path), call the "
                f"appropriate data-reading tool FIRST.\n"
                f"When your plan is ready, you MUST call the `submit_plan` tool "
                f"with the plan JSON.  Do NOT output raw JSON text.\n"
                f"If this is just a greeting or general conversation (not a CI problem), "
                f"still call `submit_plan` with problem_type='general_chat' and empty steps.\n"
                f"Consider speed/accuracy preference & problem size."
            )),
        ] + messages

    response = planner_model.invoke(messages)
    return {"messages": [response]}


def should_continue_planner_tools(state: AgentPlannerState) -> str:
    """Route: if planner made tool calls -> 'continue', else -> 'end'."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "continue"
    return "end"


# ============================================================================
# 4.  Executor sub-graph nodes  (mini ReAct)
# ============================================================================

_executor_tool_names = ", ".join(t.name for t in EXECUTOR_TOOLS)


def call_executor(state: AgentExecutorState) -> dict:
    """Invoke the executor LLM to execute one plan step via tool calls."""
    executor_model = get_executor_model()
    messages = list(state["messages"])

    if not any(isinstance(m, SystemMessage) for m in messages):
        sys_prompt = executor_sys_prompt.EXECUTOR_SYSTEM_PROMPT.format(
            tool_names=_executor_tool_names
        )
        messages = [SystemMessage(content=sys_prompt)] + messages

    response = executor_model.invoke(messages)
    return {"messages": [response]}


def should_continue_executor_tools(state: AgentExecutorState) -> str:
    """Route: if executor made tool calls -> 'continue', else -> 'end'."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "continue"
    return "end"


# ============================================================================
# 5.  Outer-graph node functions
#     planner_step & executor_step receive the compiled sub-graph as a param.
#     agent.py wraps them in closures so the StateGraph sees (state) -> dict.
# ============================================================================

def _extract_plan_from_messages(messages: List[BaseMessage]) -> Optional[dict]:
    """Walk messages backwards and find the submit_plan tool call or result."""
    # Strategy 1: look for the AIMessage that called submit_plan
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls"):
            for tc in (msg.tool_calls or []):
                if tc.get("name") == "submit_plan":
                    raw = tc.get("args", {}).get("plan_json", "")
                    try:
                        return json.loads(raw) if isinstance(raw, str) else raw
                    except (json.JSONDecodeError, TypeError):
                        pass

    # Strategy 2: look for ToolMessage from submit_plan with status "ok"
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and msg.content:
            try:
                parsed = json.loads(msg.content)
                if isinstance(parsed, dict) and parsed.get("status") == "ok" and "plan" in parsed:
                    return parsed["plan"]
            except (json.JSONDecodeError, TypeError):
                pass

    # Strategy 3 (fallback): try parsing last AI message as raw JSON
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            try:
                return _extract_json(msg.content)
            except (json.JSONDecodeError, ValueError):
                pass

    return None


def planner_step(state: PlanExecutionState, planner_graph) -> dict:
    """Run the planner sub-graph and parse its output into a Plan."""
    user_input = state["input"]

    planner_result = planner_graph.invoke({
        "messages": [],
        "user_input": user_input,
        "data_info": None,
        "plan_json": None,
    })

    plan_dict = _extract_plan_from_messages(planner_result["messages"])

    if plan_dict:
        try:
            steps = [PlanStep(**s) for s in plan_dict.get("steps", [])]
            plan = Plan(
                problem_type=plan_dict.get("problem_type", "unknown"),
                selected_method=plan_dict.get("selected_method", ""),
                reasoning=plan_dict.get("reasoning", ""),
                steps=steps,
                backup_method=plan_dict.get("backup_method"),
                confidence=plan_dict.get("confidence", 0.5),
            )
        except Exception as e:
            plan = Plan(
                problem_type="unknown",
                selected_method="unknown",
                reasoning=f"Failed to build plan from dict: {e}",
                steps=[PlanStep(step_id=1, description="Analyse the problem manually")],
                confidence=0.3,
            )
    else:
        plan = Plan(
            problem_type="unknown",
            selected_method="unknown",
            reasoning="Could not extract plan from planner output.",
            steps=[PlanStep(step_id=1, description="Analyse the problem manually")],
            confidence=0.3,
        )

    return {
        "messages": [AIMessage(content=f"Plan created: {plan.model_dump_json(indent=2)}")],
        "plan": plan,
        "current_step_index": 0,
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def executor_step(state: PlanExecutionState, executor_graph) -> dict:
    """Execute plan steps one-by-one through the executor sub-graph.

    If the plan has NO steps (general_chat, etc.), skip to replanner.
    """
    plan = state["plan"]
    current_index = state["current_step_index"]
    past_steps = list(state.get("past_steps", []))
    model_store = dict(state.get("model_store", {}))
    new_messages: List[BaseMessage] = []

    # ── Handle empty plans (general_chat, unknown, etc.) ────────────────
    if not plan or not plan.steps:
        return {
            "past_steps": past_steps,
            "current_step_index": 0,
            "model_store": model_store,
            "messages": [AIMessage(content="No execution steps -- forwarding to replanner.")],
            "should_replan": True,
        }

    # ── Execute each step ───────────────────────────────────────────────
    while current_index < len(plan.steps):
        step = plan.steps[current_index]
        args = dict(step.tool_args) if step.tool_args else {}

        # Inject stored model_id for inference steps
        if step.tool_name and step.tool_name.startswith("inference_"):
            method_key = step.tool_name.replace("inference_", "").replace("_tool", "")
            if method_key in model_store:
                args["model_id"] = model_store[method_key]

        instruction = (
            f"Execute this step:\n"
            f"Tool: {step.tool_name}\n"
            f"Description: {step.description}\n"
            f"Arguments: {json.dumps(args, default=str)}\n"
            f"Call the tool now."
        )

        exec_result = executor_graph.invoke({
            "messages": [HumanMessage(content=instruction)],
            "current_step": step.model_dump(),
            "model_store": model_store,
            "step_result": None,
        })

        # Parse result from executor messages (prefer ToolMessages)
        result: Dict[str, Any] = {}
        for msg in reversed(exec_result["messages"]):
            if hasattr(msg, "content") and msg.content:
                try:
                    parsed = (
                        json.loads(msg.content)
                        if isinstance(msg.content, str)
                        else msg.content
                    )
                    if isinstance(parsed, dict):
                        result = convert_numpy_types(parsed)
                        break
                except (json.JSONDecodeError, TypeError):
                    pass

        if not result:
            last_content = (
                exec_result["messages"][-1].content
                if exec_result["messages"]
                else ""
            )
            result = {"output": str(last_content)[:500]}

        # Track model IDs from training tools
        if isinstance(result, dict) and "model_id" in result:
            method_key = step.tool_name.replace("train_", "").replace("_tool", "")
            model_store[method_key] = result["model_id"]

        step.status = "completed" if "error" not in result else "failed"
        past_steps.append((step, result))
        new_messages.append(
            AIMessage(content=f"Executed {step.tool_name}: {str(result)[:300]}")
        )
        current_index += 1

        if step.status == "failed":
            break

    return {
        "past_steps": past_steps,
        "current_step_index": current_index,
        "model_store": model_store,
        "messages": new_messages,
        "should_replan": True,
    }


def replan_step(state: PlanExecutionState) -> dict:
    """Evaluate results and decide: continue / adjust / replan / complete."""
    plan = state["plan"]
    past_steps = state.get("past_steps", [])
    current_index = state["current_step_index"]

    all_steps_done = plan is None or current_index >= len(plan.steps)

    # ── Handle general_chat fast-path ───────────────────────────────────
    if plan and plan.problem_type == "general_chat":
        return {
            "messages": [AIMessage(content="Replanner decision: complete (general_chat)")],
            "should_replan": False,
            "final_response": (
                f"The user said: \"{state['input']}\"\n\n"
                f"Planner reasoning: {plan.reasoning}\n\n"
                "Please respond in a friendly, helpful way."
            ),
            "iteration_count": state.get("iteration_count", 0) + 1,
        }

    # ── Build evaluation context ────────────────────────────────────────
    steps_summary = "\n".join(
        f"Step {s.step_id}: {s.description}\n"
        f"  Tool: {s.tool_name} | Status: {s.status}\n"
        f"  Result: {json.dumps(r, default=str)[:300]}"
        for s, r in past_steps[-5:]
    )

    metrics_summary = ""
    for step, result in past_steps:
        if isinstance(result, dict) and "metrics" in result:
            metrics_summary += (
                f"\n{step.tool_name}: {json.dumps(result['metrics'], default=str)}"
            )

    eval_prompt = (
        f"Original problem: {state['input'][:600]}\n\n"
        f"Plan: {plan.selected_method if plan else 'None'} -- "
        f"{plan.reasoning[:300] if plan else 'N/A'}\n"
        f"Backup method: {plan.backup_method if plan else 'None'}\n\n"
        f"Completed steps:\n{steps_summary}\n\n"
        f"Metrics:\n{metrics_summary or 'None'}\n\n"
        f"Steps remaining: {len(plan.steps) - current_index if plan else 0}\n"
        f"All steps done: {all_steps_done}\n"
        f"Iteration: {state.get('iteration_count', 0)}\n\n"
        f"Decide: continue, adjust, replan, or complete?\n"
        f"If completing, provide a comprehensive Results Analysis in final_response.\n\n"
        f"You MUST call the `submit_decision` tool with your decision JSON.  "
        f"Do NOT output raw JSON text."
    )

    replan_result = _run_replanner_react(eval_prompt)

    # replan_result can be a dict (from submit_decision) or a string (fallback)
    if isinstance(replan_result, dict):
        decision_dict = replan_result
    else:
        try:
            decision_dict = json.loads(replan_result)
        except (json.JSONDecodeError, TypeError):
            try:
                decision_dict = _extract_json(replan_result)
            except Exception:
                decision_dict = {
                    "decision": "complete" if all_steps_done else "continue",
                    "final_response": replan_result if all_steps_done else None,
                }

    decision = decision_dict.get("decision", "continue")
    final_response = decision_dict.get("final_response")
    adjustments = decision_dict.get("adjustments")
    new_plan_json = decision_dict.get("new_plan")

    updated: dict = {
        "messages": [AIMessage(content=f"Replanner decision: {decision}")],
        "iteration_count": state.get("iteration_count", 0) + 1,
    }

    if decision == "complete":
        updated["should_replan"] = False
        updated["final_response"] = final_response or str(replan_result)

    elif decision == "adjust" and adjustments:
        if past_steps:
            last_step, _ = past_steps[-1]
            if last_step.tool_args:
                last_step.tool_args.update(adjustments)
            last_step.status = "pending"
        updated["current_step_index"] = max(0, current_index - 1)
        updated["plan"] = plan
        updated["should_replan"] = True
        updated["final_response"] = None

    elif decision == "replan" and new_plan_json:
        try:
            if isinstance(new_plan_json, str):
                new_plan_json = json.loads(new_plan_json)
            new_steps = [PlanStep(**s) for s in new_plan_json.get("steps", [])]
            new_plan = Plan(
                problem_type=new_plan_json.get(
                    "problem_type", plan.problem_type if plan else "unknown"
                ),
                selected_method=new_plan_json.get("selected_method", ""),
                reasoning=new_plan_json.get("reasoning", "Replanned."),
                steps=new_steps,
                backup_method=new_plan_json.get("backup_method"),
                confidence=new_plan_json.get("confidence", 0.5),
            )
            updated["plan"] = new_plan
            updated["current_step_index"] = 0
            updated["past_steps"] = []
        except Exception:
            pass
        updated["should_replan"] = True
        updated["final_response"] = None

    else:
        # "continue"
        updated["should_replan"] = True
        updated["final_response"] = None

    return updated


def _extract_decision_from_messages(messages: List[BaseMessage]) -> Optional[dict]:
    """Walk messages backwards and find submit_decision tool call or result."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls"):
            for tc in (msg.tool_calls or []):
                if tc.get("name") == "submit_decision":
                    raw = tc.get("args", {}).get("decision_json", "")
                    try:
                        return json.loads(raw) if isinstance(raw, str) else raw
                    except (json.JSONDecodeError, TypeError):
                        pass

    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and msg.content:
            try:
                parsed = json.loads(msg.content)
                if isinstance(parsed, dict) and parsed.get("status") == "ok" and "decision" in parsed:
                    return parsed["decision"]
            except (json.JSONDecodeError, TypeError):
                pass

    return None


def _run_replanner_react(eval_prompt: str) -> Any:
    """Run replanner as a mini ReAct loop.

    Returns the decision dict (from submit_decision) or raw text fallback.
    """
    replanner_model = get_replanner_model()

    messages: List[BaseMessage] = [
        SystemMessage(content=replanner_sys_prompt.REPLANNER_SYSTEM_PROMPT),
        HumanMessage(content=eval_prompt),
    ]

    for _ in range(5):
        response = replanner_model.invoke(messages)
        messages.append(response)

        if not (hasattr(response, "tool_calls") and response.tool_calls):
            # No tool calls -- return raw content (fallback)
            return response.content

        # Check if submit_decision was called -> extract and return immediately
        for tc in response.tool_calls:
            if tc.get("name") == "submit_decision":
                raw = tc.get("args", {}).get("decision_json", "")
                try:
                    return json.loads(raw) if isinstance(raw, str) else raw
                except (json.JSONDecodeError, TypeError):
                    pass

        # Execute other tool calls (data-reading tools during replan)
        tool_node = ToolNode(tools=PLANNER_TOOLS)
        tool_result = tool_node.invoke({"messages": messages})
        if isinstance(tool_result, dict) and "messages" in tool_result:
            messages.extend(tool_result["messages"])
        elif isinstance(tool_result, list):
            messages.extend(tool_result)

    # Safety fallback
    decision = _extract_decision_from_messages(messages)
    if decision:
        return decision

    return '{"decision": "complete", "final_response": "Max replanner iterations reached."}'

# ============================================================================
# 6.  Outer-graph routing
# ============================================================================

def should_replan(state: PlanExecutionState) -> str:
    """Decide whether to loop back to executor or end.

    Returns "continue" (-> agent node) or "end" (-> END).
    """
    if state.get("final_response"):
        return "end"

    if state.get("iteration_count", 0) >= 5:
        return "end"

    if state.get("should_replan", False):
        return "continue"

    return "end"