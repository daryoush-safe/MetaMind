from typing import Annotated, Optional, Sequence, TypedDict, List, Any, Dict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    """A single step in the execution plan."""
    step_id: int = Field(description="Step number in the plan")
    description: str = Field(description="What this step should accomplish")
    tool_name: Optional[str] = Field(default=None, description="Tool to use for this step")
    tool_args: Optional[Dict[str, Any]] = Field(default=None, description="Arguments for the tool")
    status: str = Field(default="pending", description="pending, in_progress, completed, failed")
    result: Optional[Any] = Field(default=None, description="Result from executing this step")
    error: Optional[str] = Field(default=None, description="Error message if step failed")


class Plan(BaseModel):
    """The complete execution plan."""
    problem_type: str = Field(description="Type of problem: tsp, classification, clustering, optimization, etc.")
    selected_method: str = Field(description="Primary CI method selected")
    reasoning: str = Field(description="Why this method was selected")
    steps: List[PlanStep] = Field(default_factory=list, description="Ordered list of execution steps")
    backup_method: Optional[str] = Field(default=None, description="Alternative method if primary fails")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence in the plan")


class ExecutionResult(BaseModel):
    """Result from executing a step or the entire plan."""
    success: bool
    method_used: str
    best_solution: Optional[Any] = None
    best_fitness: Optional[float] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    computation_time: Optional[float] = None
    recommendations: List[str] = Field(default_factory=list)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    input: str
    plan: List[str]
    current_step_index: int
    past_steps: List[tuple] # maybe using add_messages here too
    final_response: Optional[str]
    execution_result: Optional[ExecutionResult]
    iteration_count: int
    should_replan: bool
    model_store: Dict[str, str]  # e.g., {"perceptron": "model_id_123", ...}


def create_initial_state(user_input: str) -> AgentState:
    """Create initial state from user input."""
    return AgentState(
        messages=[HumanMessage(content=user_input)],
        input=user_input,
        plan=None,
        current_step_index=0,
        past_steps=[],
        final_response=None,
        execution_result=None,
        iteration_count=0,
        should_replan=False,
        model_store={}
    )
