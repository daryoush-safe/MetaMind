import os
import json
import numpy as np
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

from .state import PlanExecutionState, Plan, PlanStep, AgentExecutorState
from .prompts.planner_sys_prompt import PLANNER_SYSTEM_PROMPT
from .prompts.executor_sys_prompt import EXECUTOR_SYSTEM_PROMPT
from .prompts.replanner_sys_prompt import REPLANNER_SYSTEM_PROMPT
from .tools.tool_registry import ALL_TOOLS, DATA_LOADING_TOOLS, DATA_TOOL_OUTPUT_KEYS


# ============================================================================
# 1.  Utility helpers
# ============================================================================

def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        converted = [convert_numpy_types(item) for item in obj]
        return type(obj)(converted) if isinstance(obj, tuple) else converted
    else:
        return obj
    

def _resolve_data_references(args: dict, data_store: dict, model_store: dict) -> dict:
    """
    Resolve $DATA.<key> and $MODEL placeholders in tool_args.
    
    - "$DATA.<key>" is replaced with data_store[key]
    - "$MODEL" is NOT handled here (model_id injection is done separately for inference tools)
    
    Works recursively on nested dicts/lists in case args contain complex structures.
    """
    resolved = {}
    for k, v in args.items():
        if isinstance(v, str):
            if v.startswith("$DATA."):
                data_key = v[len("$DATA."):]
                if data_key in data_store:
                    resolved[k] = data_store[data_key]
                else:
                    print(f"WARNING: $DATA.{data_key} not found in data_store. Available keys: {list(data_store.keys())}")
                    resolved[k] = v  # keep original so the error is visible
            elif v == "$MODEL":
                # Leave for the existing model_id injection logic
                resolved[k] = v
            else:
                resolved[k] = v
        elif isinstance(v, dict):
            resolved[k] = _resolve_data_references(v, data_store, model_store)
        elif isinstance(v, list):
            resolved[k] = [
                _resolve_data_references(item, data_store, model_store) if isinstance(item, dict)
                else item
                for item in v
            ]
        else:
            resolved[k] = v
    return resolved


# ============================================================================
# 2.  Lazy LLM
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
    return _get_llm(temperature=0, model_env="PLANNER_MODEL")


def get_executor_model():
    """Return executor LLM bound to executor tools."""
    return _get_llm(temperature=0, model_env="EXECUTOR_MODEL")


def get_replanner_model():
    """Return replanner LLM bound to planner tools (for data re-reading)."""
    return _get_llm(temperature=0, model_env="MODEL_NAME")


# ============================================================================
# 3.  Executor sub-graph nodes  (mini ReAct)
# ============================================================================

def call_executor(state: AgentExecutorState) -> dict:
    tool_name = state.get("step_tool_name")
    tool = next((t for t in ALL_TOOLS if t.name == tool_name), None)
    
    messages = list(state["messages"])
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=EXECUTOR_SYSTEM_PROMPT)] + messages

    if tool is None:
        # No valid tool — just let the model respond as plain text
        executor_model = get_executor_model()
    else:
        tool_already_called = any(isinstance(m, ToolMessage) for m in messages)
        if tool_already_called:
            executor_model = get_executor_model()
        else:
            executor_model = get_executor_model().bind_tools(
                [tool], tool_choice={"type": "function", "function": {"name": tool.name}}
            )

    response = executor_model.invoke(messages)
    return {"messages": [response]}


def should_continue_executor_tools(state: AgentExecutorState) -> str:
    tool_name = state.get("step_tool_name")
    tool = next((t for t in ALL_TOOLS if t.name == tool_name), None)
    
    if tool is None:
        return "end"
    
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "continue"
    return "end"


# ============================================================================
# 5.  Outer-graph node functions
# ============================================================================

def plan_step(state: PlanExecutionState) ->PlanExecutionState:
    """
    Analyze the problem and create an execution plan.
    
    This node:
    1. Receives user input describing the problem
    2. Classifies the problem type
    3. Selects appropriate CI method(s)
    4. Creates step-by-step execution plan
    """
    user_input = state["input"]

    planner_llm = get_planner_model()

    message = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=f"""Analyze this problem and create an execution plan:

{user_input}

Remember to output a valid JSON plan.
Consider:
1. Does the user mention any preference for speed vs accuracy?
2. Is ground truth data available for computing metrics?
3. What are the appropriate parameters based on problem size and user preferences?
4. Does the user reference any external files (CSV, TSP, etc.)? If so, plan a data-loading step first and use $DATA references in subsequent steps.""")
    ]

    history = state["messages"][-10:-1]
    response = planner_llm.invoke(message + history)

    try:
        response_text = response.content
            
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0]
        else:
            json_str = response_text
            
        plan_dict = json.loads(json_str.strip())
        
        steps = [PlanStep(**step) for step in plan_dict.get("steps", [])]
        plan = Plan(
            problem_type=plan_dict.get("problem_type", "unknown"),
            selected_method=plan_dict.get("selected_method", ""),
            reasoning=plan_dict.get("reasoning", ""),
            steps=steps,
            backup_method=plan_dict.get("backup_method"),
            confidence=plan_dict.get("confidence", 0.5)
        )

    except (json.JSONDecodeError, KeyError) as e:
        plan = Plan(
            problem_type="unknown",
            selected_method="unknown",
            reasoning=f"Failed to parse plan: {str(e)}. Using default approach.",
            steps=[
                PlanStep(
                    step_id=1,
                    description="Analyze the problem manually",
                    tool_name=None,
                    tool_args=None
                )
            ],
            confidence=0.3
        )
    new_messages = [
        HumanMessage(content=user_input),
        AIMessage(content=f"Created plan: {plan.model_dump_json(indent=2)}")
    ]
    
    return {
        "messages": new_messages,
        "plan": plan,
        "current_step_index": 0,
        "iteration_count": state.get("iteration_count", 0) + 1
    }


def execute_step(state: PlanExecutionState, executor_graph) -> PlanExecutionState:
    plan = state["plan"]
    current_index = state["current_step_index"]
    past_steps = list(state.get("past_steps", []))
    new_model_store = dict(state.get("model_store", {}))
    new_data_store = dict(state.get("data_store", {}))
    new_messages = []

    if not plan or not plan.steps:
        return {
            "messages": [AIMessage(content="No execution steps -- forwarding to replanner.")],
            "should_replan": True,
        }
    
    while current_index < len(plan.steps):
        step = plan.steps[current_index]
        tool_to_use = next((t for t in ALL_TOOLS if t.name == step.tool_name), None)
        args = step.tool_args.copy() if step.tool_args else {}
        args = _resolve_data_references(args, new_data_store, new_model_store)

        # ---- Inject model_id for inference tools ----
        if tool_to_use and step.tool_name.startswith("inference_"):
            method_key = step.tool_name.replace('inference_', '').replace('_tool', '')
            
            if method_key in new_model_store:
                args["model_id"] = new_model_store[method_key]
            elif args.get("model_id") == "$MODEL":
                print(f"WARNING: No model_id found in store for {method_key}")
            # else: model_id might already be set explicitly

        # Remove any remaining "$MODEL" string if it wasn't resolved
        if args.get("model_id") == "$MODEL":
            method_key = step.tool_name.replace('inference_', '').replace('_tool', '')
            if method_key in new_model_store:
                args["model_id"] = new_model_store[method_key]
            else:
                print(f"WARNING: $MODEL reference unresolved for {step.tool_name}")


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
            "model_store": new_model_store,
            "step_tool_name": step.tool_name,
            "step_result": None,
        })

        if step.status != "failed":
            step.status = "completed"

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
        
        # ---- Store model_id from training tools ----
        if isinstance(result, dict) and 'model_id' in result:
            method_key = step.tool_name.replace('train_', '').replace('_tool', '')
            new_model_store[method_key] = result['model_id']

        # ---- Store data from data-loading tools ----
        if step.tool_name in DATA_LOADING_TOOLS and isinstance(result, dict):
            expected_keys = DATA_TOOL_OUTPUT_KEYS.get(step.tool_name, [])
            for key in expected_keys:
                if key in result:
                    new_data_store[key] = result[key]
            # Also store any extra keys the tool might return
            for key, value in result.items():
                if key not in new_data_store:
                    new_data_store[key] = value
            print(f"Data store updated with keys: {list(new_data_store.keys())}")

        past_steps.append((step, result))
        new_messages.append(AIMessage(content=f"Executed {step.tool_name or 'LLM'}: {str(result)[:200]}"))
        current_index += 1

        if step.status == "failed":
            break

    return {
        "past_steps": past_steps,
        "current_step_index": current_index,
        "model_store": new_model_store,
        "data_store": new_data_store,
        "messages": new_messages,
        "should_replan": True 
    }


def replan_step(state: PlanExecutionState) -> PlanExecutionState:
    """
    Evaluate results and decide whether to continue, adjust, or complete.
    
    This node:
    1. Reviews execution results
    2. Decides if plan needs adjustment
    3. Generates final response if complete
    """
    plan = state["plan"]
    past_steps = state.get("past_steps", [])
    current_index = state["current_step_index"]
    
    # Check if all steps are complete
    all_steps_done = plan is None or current_index >= len(plan.steps)
    
    # Build evaluation prompt
    steps_summary = "\n".join([
        f"Step {step.step_id}: {step.description}\nResult: {json.dumps(result, default=str)[:200]}"
        for step, result in past_steps[-3:]  # Last 3 steps
    ])

    metrics_summary = ""
    for step, result in past_steps:
        if isinstance(result, dict) and 'metrics' in result:
            metrics_summary += f"\nMetrics from {step.tool_name}: {json.dumps(result['metrics'], default=str)}"
    
    eval_prompt = f"""Evaluate the execution progress:

Original Problem: {state['input'][:500]}

Plan: {plan.selected_method if plan else 'None'} - {plan.reasoning[:200] if plan else 'No plan'}

Completed Steps:
{steps_summary}

Extracted Metrics:
{metrics_summary if metrics_summary else "No metrics available"}

Steps remaining: {len(plan.steps) - current_index if plan else 0}
All steps done: {all_steps_done}

Decide: continue, adjust, replan, or complete?

If completing, provide a comprehensive Results Analysis following the format in your instructions."""

    replanner_llm = get_replanner_model()
    messages = [
        SystemMessage(content=REPLANNER_SYSTEM_PROMPT),
        HumanMessage(content=eval_prompt)
    ]

    # history = state["messages"][-10:-1]
    # response = replanner_llm.invoke(messages + history)
    response = replanner_llm.invoke(messages)
    
    try:
        response_text = response.content
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0]
        else:
            json_str = response_text
            
        decision_dict = json.loads(json_str.strip())
        decision = decision_dict.get("decision", "continue")
        final_response = decision_dict.get("final_response")
        
    except (json.JSONDecodeError, KeyError):
        # Default to continue if parsing fails
        decision = "complete" if all_steps_done else "continue"
        final_response = response.content if all_steps_done else None
    
    # Update state based on decision
    should_continue = decision in ["continue", "adjust"]
    
    new_messages = [AIMessage(content=f"Replanner decision: {decision}")]
    
    return {
        "messages": new_messages,
        "should_replan": should_continue,
        "final_response": final_response,
        "iteration_count": state.get("iteration_count", 0) + 1
    }

def should_continue(state: PlanExecutionState) -> str:
    """
    decide whether to continue execution or end.
    Returns True to continue (go back to agent), False to end.
    """
    if state.get("final_response"):
        return "end"
    
    if state.get("iteration_count", 0) >= 5:
        return "end"

    if state.get("should_replan", False):
        return "continue"

    return "end"