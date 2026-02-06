import json
from langchain_core.tools import tool

@tool
def submit_decision(decision_json: str) -> str:
    """Submit the replanner's evaluation decision.

    Call this tool when you have finished evaluating the execution results.
    Pass the complete decision as a JSON string.

    Parameters
    ----------
    decision_json : str
        JSON string with keys: decision, reasoning, adjustments, new_plan,
        final_response, recommendations, performance_assessment.
    """
    try:
        parsed = json.loads(decision_json)
        return json.dumps({"status": "ok", "decision": parsed})
    except json.JSONDecodeError as e:
        return json.dumps({"status": "error", "message": f"Invalid JSON: {e}"})