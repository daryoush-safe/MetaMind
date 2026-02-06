import json
from langchain_core.tools import tool

@tool
def submit_plan(plan_json: str) -> str:
    """Submit the final execution plan.

    Call this tool AFTER you have finished reading/preprocessing any data.
    Pass the complete plan as a JSON string.

    Parameters
    ----------
    plan_json : str
        The plan as a JSON string with keys: problem_type, selected_method,
        reasoning, user_preference, steps, backup_method, confidence,
        expected_metrics.
    """
    try:
        parsed = json.loads(plan_json)
        return json.dumps({"status": "ok", "plan": parsed})
    except json.JSONDecodeError as e:
        return json.dumps({"status": "error", "message": f"Invalid JSON: {e}"})