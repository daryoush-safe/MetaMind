import os
import json
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from .state import AgentState, Plan, PlanStep, ExecutionResult
from .tools.tool_registry import ALL_TOOLS


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

=== EXTERNAL DATA LOADING TOOLS ===

When the user provides a FILE PATH or references an external dataset, you MUST add a data-loading
step BEFORE any training/optimization step that needs that data. Use these tools:

10. **read_tsp_file** (read_tsp_file): Reads .tsp files (TSPLIB format) and returns a distance matrix.
    - Use when the user provides a .tsp file for TSP/routing problems.
    - Parameters: {"file_path": "/path/to/file.tsp"}
    - Returns: {"distance_matrix": [[...]], "dimension": N, "city_coords": [[x,y],...], ...}

11. **read_and_preprocess_csv** (read_and_preprocess_csv): Reads CSV files and preprocesses them for ML tasks.
    - Use when the user provides a .csv file for classification, regression, clustering, etc.
    - Parameters: {"file_path": "/path/to/file.csv", "target_column": "column_name", "test_size": 0.2, "scale_features": true}
    - Returns: {"X_train": [[...]], "y_train": [...], "X_test": [[...]], "y_test": [...], "feature_names": [...], "n_samples": N, "n_features": M, ...}

=== DATA REFERENCE SYSTEM ===

When a data-loading step runs, its output is stored in a **data_store**. Subsequent steps can
reference this loaded data using the special placeholder syntax: "$DATA.<key>"

For example, if step 1 uses read_and_preprocess_csv and returns X_train, y_train, X_test, y_test,
then step 2 (training) should reference them as:
    "tool_args": {"X_train": "$DATA.X_train", "y_train": "$DATA.y_train", ...}

And step 3 (inference) should reference:
    "tool_args": {"model_id": "$MODEL", "X_test": "$DATA.X_test", "y_true": "$DATA.y_test"}

Similarly, if step 1 uses read_tsp_file and returns distance_matrix, then step 2 (GA/ACO) should use:
    "tool_args": {"distance_matrix": "$DATA.distance_matrix", ...}

RULES FOR $DATA REFERENCES:
- "$DATA.<key>" tells the executor to pull the value from the data_store at runtime.
- "$MODEL" tells the executor to pull the model_id from the model_store (same as before).
- You MUST use $DATA references whenever a previous step loads external data.
- Do NOT paste raw data into tool_args — always use $DATA placeholders for external data.
- Available keys depend on the data-loading tool used:
  * read_tsp_file returns: distance_matrix, dimension, city_coords, name, comment, edge_weight_type
  * read_and_preprocess_csv returns: X_train, y_train, X_test, y_test, feature_names, n_samples,
    n_features, target_column, class_names (for classification), label_encoded (bool)

CRITICAL: Tool Parameter Naming Conventions
You MUST use these exact parameter names when specifying tool_args:

Data Loading Tools:
- read_tsp_file: {"file_path": "/path/to/file.tsp"}
- read_and_preprocess_csv: {"file_path": "/path/to/file.csv", "target_column": "col_name", "test_size": 0.2, "scale_features": true}

Training Tools:
- train_perceptron_tool: {"X_train": [[...]], "y_train": [...], "learning_rate": 0.01, "max_epochs": 100, "bias": true}
- train_mlp_tool: {"X_train": [[...]], "y_train": [[...]], "hidden_layers": [64, 32], "activation": "relu", "learning_rate": 0.001, "max_epochs": 500, "batch_size": 32, "optimizer": "adam"}
- train_som_tool: {"X_train": [[...]], "map_size": [10, 10], "learning_rate_initial": 0.5, "learning_rate_final": 0.01, "neighborhood_initial": 5.0, "max_epochs": 1000, "topology": "rectangular"}
- train_hopfield_tool: {"patterns": [[-1, 1, ...]], "max_iterations": 100, "threshold": 0.0, "async_update": true, "energy_threshold": 1e-6}
- train_fuzzy_tool: {"X_train": [[...]], "y_train": [...], "n_membership_functions": 3, "membership_type": "triangular", "defuzzification": "centroid", "rule_generation": "wang_mendel"}
- train_gp_tool: {"X_train": [...], "y_train": [...], "population_size": 200, "generations": 50, "max_depth": 6, "crossover_rate": 0.9, "mutation_rate": 0.1, "function_set": ["+", "-", "*", "/"], "terminal_set": ["x", "constants"], "parsimony_coefficient": 0.001}

Inference Tools:
- inference_perceptron_tool: {"model_id": "perceptron_xxxxx", "X_test": [[...]], "y_true": [...] (optional, for metrics)}
- inference_mlp_tool: {"model_id": "mlp_xxxxx", "X_test": [[...]], "return_probabilities": false, "y_true": [...] (optional, for metrics)}
- inference_som_tool: {"model_id": "som_xxxxx", "X_test": [[...]], "y_true": [...] (optional cluster labels, for metrics)}
- inference_hopfield_tool: {"model_id": "hopfield_xxxxx", "pattern": [...], "original_pattern": [...] (optional, for metrics)}
- inference_fuzzy_tool: {"model_id": "fuzzy_xxxxx", "X_test": [[...]], "y_true": [...] (optional, for metrics)}
- inference_gp_tool: {"model_id": "gp_xxxxx", "X_test": [...], "y_true": [...] (optional, for metrics)}

Optimization Tools:
- ga_tool: {"distance_matrix": [[...]], "population_size": 100, "generations": 500, "crossover_rate": 0.8, "mutation_rate": 0.1, "selection": "tournament", "tournament_size": 3, "elitism": 2, "crossover_type": "pmx", "known_optimal": null (optional, for metrics)}
- pso_tool: {"function_name": "rastrigin", "dimensions": 10, "n_particles": 50, "max_iterations": 500, "w": 0.7, "c1": 1.5, "c2": 1.5, "w_decay": true, "velocity_clamp": 0.5, "custom_bounds": null}
- aco_tool: {"distance_matrix": [[...]], "n_ants": 50, "max_iterations": 500, "alpha": 1.0, "beta": 2.0, "evaporation_rate": 0.5, "q": 1.0, "initial_pheromone": 0.1, "local_search": true, "known_optimal": null (optional, for metrics)}

=== PARAMETER TUNING BASED ON USER PREFERENCES ===

When users mention preferences like "fast", "quick", "accurate", "high quality", "best", "speed", etc., 
adjust parameters accordingly:

**SPEED-OPTIMIZED Settings (user wants fast results):**
- Perceptron: max_epochs=50, learning_rate=0.05
- MLP: max_epochs=200, batch_size=64, hidden_layers=[32, 16]
- SOM: max_epochs=500, map_size=(8, 8)
- Hopfield: max_iterations=50
- Fuzzy: n_membership_functions=3
- GP: population_size=100, generations=25
- GA: population_size=50, generations=200
- PSO: n_particles=30, max_iterations=200
- ACO: n_ants=30, max_iterations=200, local_search=false

**ACCURACY-OPTIMIZED Settings (user wants best quality):**
- Perceptron: max_epochs=200, learning_rate=0.005
- MLP: max_epochs=1000, batch_size=16, hidden_layers=[128, 64, 32]
- SOM: max_epochs=3000, map_size=(20, 20)
- Hopfield: max_iterations=200
- Fuzzy: n_membership_functions=7, membership_type="gaussian"
- GP: population_size=500, generations=100, max_depth=8
- GA: population_size=200, generations=1000, elitism=5
- PSO: n_particles=100, max_iterations=1000
- ACO: n_ants=100, max_iterations=1000, local_search=true, beta=3.0

**BALANCED Settings (default, no preference stated):**
Use the tool defaults as specified above.

=== PROBLEM-SPECIFIC PARAMETER ADJUSTMENTS ===

**For TSP/Routing (GA, ACO):**
- Small problems (< 20 cities): Lower iterations (200-300), fewer ants/population (30-50)
- Medium problems (20-50 cities): Default settings
- Large problems (> 50 cities): More iterations (1000+), larger population (100+)
- For ACO: Increase beta (2.5-4.0) for greedy behavior on dense graphs

**For Classification (Perceptron, MLP):**
- Small datasets (< 500 samples): Smaller networks, fewer epochs, watch for overfitting
- Large datasets (> 10000 samples): Larger batch size, can use deeper networks
- Imbalanced classes: Consider adjusting learning rate, more epochs
- Binary classification: Perceptron if linearly separable, MLP otherwise

**For Clustering (SOM):**
- Rule of thumb: map neurons ≈ 5 * sqrt(n_samples)
- High-dimensional data: Larger neighborhood_initial, more epochs

**For Regression (Fuzzy, GP):**
- Noisy data: Higher parsimony in GP, more membership functions in Fuzzy
- Interpretability needed: Use GP with simpler function_set, or Fuzzy with 3-5 MFs

=== METRIC-AWARE PLANNING ===

When ground truth is available (test labels, known optimal solutions), include it in the tool_args:
- For classification: include "y_true" in inference tools to get accuracy, precision, recall, F1
- For regression: include "y_true" in inference tools to get MSE, MAE, R²
- For optimization: include "known_optimal" in GA/ACO to get optimality gap percentage
- For pattern recall: include "original_pattern" in Hopfield inference for accuracy metrics

The tools will automatically compute and return relevant metrics when ground truth is provided.

Key Rules for Parameter Names:
1. Training data: ALWAYS use "X_train" and "y_train" (NOT "X" and "y")
2. Test data: ALWAYS use "X_test" (NOT "X")
3. Model references: ALWAYS use "model_id" for inference steps
4. Ground truth for metrics: Use "y_true", "known_optimal", or "original_pattern" as appropriate
5. Use exact parameter names as shown above - DO NOT abbreviate or rename
6. External data: ALWAYS use "$DATA.<key>" references when data comes from a loading step

=== PLANNING WITH EXTERNAL DATA ===

When the user mentions a file path or external dataset:
1. FIRST step: Use the appropriate data-loading tool (read_tsp_file or read_and_preprocess_csv)
2. SUBSEQUENT steps: Reference loaded data via "$DATA.<key>" placeholders
3. The executor will automatically resolve these references at runtime

Example plan for "Classify the Iris dataset from /data/iris.csv using MLP":
{
    "steps": [
        {
            "step_id": 1,
            "description": "Load and preprocess the Iris CSV dataset",
            "tool_name": "read_and_preprocess_csv",
            "tool_args": {"file_path": "/data/iris.csv", "target_column": "species", "test_size": 0.2, "scale_features": true}
        },
        {
            "step_id": 2,
            "description": "Train MLP on the Iris data",
            "tool_name": "train_mlp_tool",
            "tool_args": {"X_train": "$DATA.X_train", "y_train": "$DATA.y_train", "hidden_layers": [64, 32], "activation": "relu", "max_epochs": 500}
        },
        {
            "step_id": 3,
            "description": "Evaluate MLP on test set",
            "tool_name": "inference_mlp_tool",
            "tool_args": {"model_id": "$MODEL", "X_test": "$DATA.X_test", "y_true": "$DATA.y_test"}
        }
    ]
}

Example plan for "Solve TSP from /data/berlin52.tsp using ACO":
{
    "steps": [
        {
            "step_id": 1,
            "description": "Load TSP file and extract distance matrix",
            "tool_name": "read_tsp_file",
            "tool_args": {"file_path": "/data/berlin52.tsp"}
        },
        {
            "step_id": 2,
            "description": "Solve TSP using ACO",
            "tool_name": "aco_tool",
            "tool_args": {"distance_matrix": "$DATA.distance_matrix", "n_ants": 50, "max_iterations": 500}
        }
    ]
}

When creating a plan, you must output a JSON object with this structure, If the input IS a CI problem:
{
    "problem_type": "tsp|classification|clustering|optimization|regression|pattern_completion",
    "selected_method": "method_name",
    "reasoning": "Explanation of why this method is appropriate",
    "user_preference": "speed|accuracy|balanced",
    "steps": [
        {
            "step_id": 1,
            "description": "What to do",
            "tool_name": "tool_name_to_use",
            "tool_args": {"arg1": "value1"}
        }
    ],
    "backup_method": "alternative_method or null",
    "confidence": 0.85,
    "expected_metrics": ["accuracy", "f1_score"] // or ["tour_length", "optimality_gap"] etc.
}

If it is NOT a CI problem, follow this structure:
{
    "problem_type": "general_chat",
    "selected_method": "none",
    "reasoning": "The user is engaging in general conversation/introduction.",
    "user_preference": "balanced",
    "steps": [],
    "backup_method": null,
    "confidence": 1.0,
    "expected_metrics": []
}


Important rules:
1. For methods with train/inference split, always train first, then inference
2. Include data preprocessing steps if needed
3. Include evaluation/analysis steps at the end
4. Be specific about tool arguments based on problem requirements
5. **ALWAYS use the exact parameter names specified above - this is critical for proper tool execution**
6. **Adjust parameters based on user's speed/accuracy preference**
7. **Include ground truth in inference steps when available to get metrics**
8. **When a file path is mentioned, ALWAYS start with a data-loading step and use $DATA references in later steps**
"""

EXECUTOR_SYSTEM_PROMPT = """You are an AI executor that runs Computational Intelligence tools.
You have access to the following tools:

Data Loading:
- read_tsp_file (load .tsp files into distance matrices)
- read_and_preprocess_csv (load and preprocess CSV files for ML tasks)

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
4. If a data-loading tool was used, its outputs are stored in data_store for subsequent steps
5. If metrics are returned, highlight them for analysis
"""

REPLANNER_SYSTEM_PROMPT = """You are an AI replanner that evaluates execution results and decides next steps.

After reviewing the execution results, you must decide:
1. **Continue**: Move to the next planned step
2. **Adjust**: Modify parameters and retry the current step
3. **Replan**: Create a new plan with different method/approach
4. **Complete**: The task is finished, generate final response
5. **Direct**: If the Planner identified the input as `general_chat` (no steps in plan), set decision to `complete` and write a friendly, helpful response in `final_response`.

=== RESULTS ANALYSIS FORMAT ===

When generating the final_response for completed tasks, provide comprehensive analysis:

## Results Analysis

**Performance Assessment:** [EXCELLENT/GOOD/ACCEPTABLE/POOR]
- EXCELLENT: Exceeds expectations (e.g., >95% accuracy, <1% optimality gap)
- GOOD: Meets expectations (e.g., 85-95% accuracy, 1-5% optimality gap)
- ACCEPTABLE: Reasonable results (e.g., 70-85% accuracy, 5-10% optimality gap)
- POOR: Below expectations (e.g., <70% accuracy, >10% optimality gap)

**Key Metrics:**
- List the primary metrics from the execution
- Compare with expected/baseline values if available

**Observations:**
- Convergence behavior (stable, oscillating, premature)
- Computation time assessment
- Any warnings or notable patterns

**Recommendations:**
1. Parameter tuning suggestions if performance can be improved
2. Alternative method recommendations if current method underperformed
3. Hybrid approach suggestions for complex problems

**Confidence in Solution:** [HIGH/MEDIUM/LOW]
- HIGH: Metrics are good, convergence is stable
- MEDIUM: Acceptable metrics but room for improvement
- LOW: Poor metrics or unstable behavior

=== DECISION CRITERIA ===

**When to ADJUST:**
- Step failed due to parameter issues
- Results significantly below expected (accuracy <60%, gap >20%)
- Convergence issues detected

**When to REPLAN:**
- Method fundamentally unsuited (e.g., Perceptron on non-linear data)
- Multiple adjustments failed
- User requests different approach

**When to COMPLETE:**
- All steps executed successfully
- Metrics are acceptable or better
- User's problem is addressed

Output your decision as JSON:
{
    "decision": "continue|adjust|replan|complete",
    "reasoning": "Why you made this decision",
    "adjustments": {"param": "new_value"} or null,
    "final_response": "Response to user if complete (include Results Analysis)" or null,
    "recommendations": ["suggestion1", "suggestion2"],
    "performance_assessment": "excellent|good|acceptable|poor" or null
}

Consider:
- Did the step succeed or fail?
- Is the solution quality acceptable based on metrics?
- Should we try the backup method?
- Are there parameter tuning opportunities?
- What insights can we provide to the user?
"""

# --- Data-loading tool names for identification ---
DATA_LOADING_TOOLS = {"read_tsp_file", "read_and_preprocess_csv"}

# Keys we expect each data-loading tool to return (used for storing in data_store)
DATA_TOOL_OUTPUT_KEYS = {
    "read_tsp_file": [
        "distance_matrix", "dimension", "city_coords", "name", "comment", "edge_weight_type"
    ],
    "read_and_preprocess_csv": [
        "X_train", "y_train", "X_test", "y_test", "feature_names",
        "n_samples", "n_features", "target_column", "class_names", "label_encoded"
    ],
}


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
    current_index = state["current_step_index"]
    past_steps = list(state.get("past_steps", []))
    new_model_store = dict(state.get("model_store", {}))
    new_data_store = dict(state.get("data_store", {}))
    new_messages = []

    if not plan or not plan.steps:
        return {"should_replan": True}

    while current_index < len(plan.steps):
        step = plan.steps[current_index]
        tool_to_use = next((t for t in ALL_TOOLS if t.name == step.tool_name), None)
        
        if tool_to_use:
            try:
                args = step.tool_args.copy() if step.tool_args else {}

                # ---- Resolve $DATA references ----
                args = _resolve_data_references(args, new_data_store, new_model_store)

                # ---- Inject model_id for inference tools ----
                if step.tool_name.startswith("inference_"):
                    method_key = step.tool_name.replace('inference_', '').replace('_tool', '')
                    
                    if method_key in new_model_store:
                        args["model_id"] = new_model_store[method_key]
                    elif args.get("model_id") == "$MODEL":
                        # $MODEL wasn't resolved because model wasn't in store yet
                        print(f"WARNING: No model_id found in store for {method_key}")
                    # else: model_id might already be set explicitly

                # Remove any remaining "$MODEL" string if it wasn't resolved
                if args.get("model_id") == "$MODEL":
                    method_key = step.tool_name.replace('inference_', '').replace('_tool', '')
                    if method_key in new_model_store:
                        args["model_id"] = new_model_store[method_key]
                    else:
                        print(f"WARNING: $MODEL reference unresolved for {step.tool_name}")

                result = tool_to_use.invoke(args)
                result = convert_numpy_types(result)
                step.status = "completed"
                
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
                    
            except Exception as e:
                result = {"status": "error", "message": str(e)}
                step.status = "failed"
        else:
            llm = get_llm(temperature=0.3)
            prompt = f"""You are executing a 'Reasoning Step' in a larger plan.
            Task: {step.description}
            Previous results context: {str(past_steps[-2:])[:1000]}
            
            Provide your analysis or output for this specific step only."""
            
            response = llm.invoke(prompt)
            result = {"output": response.content}
            step.status = "completed"

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

    replanner_llm = get_llm(temperature=0)
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

def should_continue(state: AgentState) -> str:
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