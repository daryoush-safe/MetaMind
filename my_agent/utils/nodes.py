import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from .state import AgentState, Plan, PlanStep, ExecutionResult
from .tools.tool_registry import ALL_TOOLS

def get_llm(temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(
        model=os.environ.get("MODEL_NAME"),
        openai_api_key=os.environ.get("API_KEY"),
        openai_api_base=os.environ.get("API_BASE_URL"),
        temperature=temperature,
    )

# System prompts
PLANNER_SYSTEM_PROMPT = """You are an expert AI planner for Computational Intelligence problems.
Your job is to analyze the user's problem and create a detailed execution plan.

Available CI Methods and their use cases:
1. **Perceptron** (train_perceptron_tool, inference_perceptron_tool): Binary classification, linearly separable data
2. **MLP** (train_mlp_tool, inference_mlp_tool): Multi-class classification, complex patterns, Titanic-like datasets
3. **SOM** (train_som_tool, inference_som_tool): Clustering, visualization, customer segmentation
4. **Hopfield** (train_hopfield_tool, inference_hopfield_tool): Pattern completion, associative memory
5. **Fuzzy** (train_fuzzy_tool, inference_fuzzy_tool): Control systems, regression with interpretable rules
6. **GP** (train_gp_tool, inference_gp_tool): Symbolic regression, discovering mathematical formulas
7. **GA** (ga_tool): TSP, combinatorial optimization, permutation problems
8. **PSO** (pso_tool): Continuous function optimization (Rastrigin, Ackley, Rosenbrock, Sphere)
9. **ACO** (aco_tool): TSP, routing problems, graph-based optimization

Problem Type to Method Mapping:
- TSP/Routing → ACO (primary), GA (backup)
- Continuous Optimization → PSO
- Classification → MLP (complex), Perceptron (simple/binary)
- Clustering → SOM
- Pattern Recognition → Hopfield
- Control/Regression → Fuzzy, GP
- Symbolic Regression → GP

When creating a plan, you must output a JSON object with this structure, If the input IS a CI problem:
{
    "problem_type": "tsp|classification|clustering|optimization|regression|pattern_completion",
    "selected_method": "method_name",
    "reasoning": "Explanation of why this method is appropriate",
    "steps": [
        {
            "step_id": 1,
            "description": "What to do",
            "tool_name": "tool_name_to_use",
            "tool_args": {"arg1": "value1"}
        }
    ],
    "backup_method": "alternative_method or null",
    "confidence": 0.85
}

If it is NOT a CI problem, follow this structure:
{
    "problem_type": "general_chat",
    "selected_method": "none",
    "reasoning": "The user is engaging in general conversation/introduction.",
    "steps": [],
    "backup_method": null,
    "confidence": 1.0
}


Important rules:
1. For methods with train/inference split, always train first, then inference
2. Include data preprocessing steps if needed
3. Include evaluation/analysis steps at the end
4. Be specific about tool arguments based on problem requirements
"""

EXECUTOR_SYSTEM_PROMPT = """You are an AI executor that runs Computational Intelligence tools.
You have access to the following tools:

Neural Networks:
- train_perceptron_tool, inference_perceptron_tool
- train_mlp_tool, inference_mlp_tool  
- train_som_tool, inference_som_tool
- train_hopfield_tool, inference_hopfield_tool

Fuzzy Systems:
- train_fuzzy_tool, inference_fuzzy_tool

Evolutionary/Genetic:
- train_gp_tool, inference_gp_tool
- ga_tool (single tool for optimization)

Swarm Intelligence:
- pso_tool (continuous optimization)
- aco_tool (TSP/routing)

Execute the current step according to the plan. After execution:
1. Report the results clearly
2. Note any issues or unexpected outcomes
3. Save model_id if a training tool was used (for later inference)
"""

REPLANNER_SYSTEM_PROMPT = """You are an AI replanner that evaluates execution results and decides next steps.

After reviewing the execution results, you must decide:
1. **Continue**: Move to the next planned step
2. **Adjust**: Modify parameters and retry the current step
3. **Replan**: Create a new plan with different method/approach
4. **Complete**: The task is finished, generate final response
5. **Direct**: If the Planner identified the input as `general_chat` (no steps in plan), set decision to `complete` and write a friendly, helpful response in `final_response`.

Output your decision as JSON:
{
    "decision": "continue|adjust|replan|complete",
    "reasoning": "Why you made this decision",
    "adjustments": {"param": "new_value"} or null,
    "final_response": "Response to user if complete" or null,
    "recommendations": ["suggestion1", "suggestion2"]
}

Consider:
- Did the step succeed or fail?
- Is the solution quality acceptable?
- Should we try the backup method?
- Are there parameter tuning opportunities?
"""


def plan_step(state: AgentState) ->AgentState:
    """
    Analyze the problem and create an execution plan.
    
    This node:
    1. Receives user input describing the problem
    2. Classifies the problem type
    3. Selects appropriate CI method(s)
    4. Creates step-by-step execution plan
    """
    user_input = state["input"]

    planner_llm = get_llm(temperature=0)

    message = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=f"""Analyze this problem and create an execution plan:

{user_input}

Remember to output a valid JSON plan.""")
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
        # Fallback: create a simple plan
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
    
    # Update state
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


def execute_step(state: AgentState) -> AgentState:
    plan = state["plan"]
    # Start from where we left off (usually 0)
    current_index = state["current_step_index"]
    past_steps = list(state.get("past_steps", []))
    new_model_store = dict(state.get("model_store", {}))
    new_messages = []

    if not plan or not plan.steps:
        return {"should_replan": True}

    while current_index < len(plan.steps):
        step = plan.steps[current_index]
        
        tool_to_use = next((t for t in ALL_TOOLS if t.name == step.tool_name), None)
        
        if tool_to_use:
            try:
                # Inject model_id if this is an inference step
                args = step.tool_args or {}

                result = tool_to_use.invoke(args)
                step.status = "completed"
                
                if isinstance(result, dict) and 'model_id' in result:
                    method_key = step.tool_name.replace('train_', '').replace('_tool', '')
                    new_model_store[method_key] = result['model_id']
            except Exception as e:
                result = {"status": "error", "message": str(e)}
                step.status = "failed"
        else:
            llm = get_llm(temperature=0.3)
            prompt = f"""You are executing a 'Reasoning Step' in a larger plan.
            Task: {step.description}
            Context from previous steps: {state.get('past_steps', [])[-2:]}
            
            Provide your analysis or output for this specific step only."""
            
            response = llm.invoke(prompt)
            result = {"output": response.content}
            step.status = "completed"

        past_steps.append((step, result))
        new_messages.append(AIMessage(content=f"Executed {step.tool_name}: {str(result)[:200]}"))
        current_index += 1

        if step.status == "failed":
            break

    return {
        "past_steps": past_steps,
        "current_step_index": current_index,
        "model_store": new_model_store,
        "messages": new_messages,
        "should_replan": True # Always check with replanner after the batch
    }


def replan_step(state: AgentState) -> AgentState:
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
    
    eval_prompt = f"""Evaluate the execution progress:

Original Problem: {state['input'][:500]}

Plan: {plan.selected_method if plan else 'None'} - {plan.reasoning[:200] if plan else 'No plan'}

Completed Steps:
{steps_summary}

Steps remaining: {len(plan.steps) - current_index if plan else 0}
All steps done: {all_steps_done}

Decide: continue, adjust, replan, or complete?"""

    replanner_llm = get_llm(temperature=0)
    messages = [
        SystemMessage(content=REPLANNER_SYSTEM_PROMPT),
        HumanMessage(content=eval_prompt)
    ]

    history = state["messages"][-10:-1]
    response = replanner_llm.invoke(messages + history)
    
    # Parse decision
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

def should_continue(state: AgentState) -> bool:
    """
    decide whether to continue execution or end.
    Returns True to continue (go back to agent), False to end.
    """
    if state.get("final_response"):
        return False
    
    if state.get("iteration_count", 0) >= 5:
        return False

    return state.get("should_replan", False)