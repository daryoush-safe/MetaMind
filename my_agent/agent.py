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
            True: "agent",
            False: END
        }
    )
    return graph

def visualize_graph(app):
    try:
        display(Image(app.get_graph(xray=True).draw_mermaid_png()))
    except Exception as e:
        print(f"Could not visualize: {e}")

def run_interactive(app):
    """
    Run the MetaMind agent in interactive mode with persistent memory.
    """
    print("=" * 60)
    print("  MetaMind: LLM-Orchestrated Computational Intelligence")
    print("=" * 60)
    print()
    print("Available problem types:")
    print("  - TSP / Routing problems (ACO, GA)")
    print("  - Function optimization (PSO)")
    print("  - Classification (MLP, Perceptron)")
    print("  - Clustering (SOM)")
    print("  - Pattern completion (Hopfield)")
    print("  - Control / Regression (Fuzzy)")
    print("  - Symbolic regression (GP)")
    print()
    print("Type 'exit' or 'quit' to stop.")
    print("Type 'history' to see conversation history.")
    print("Type 'reset' to clear history and start fresh.")
    print("=" * 60)
    print()
    
    thread_id = "interactive_session"
    config = {
        "recursion_limit": 50,
        "configurable": {"thread_id": thread_id}
    }
    
    while True:
        try:
            user_input = input("\n🧠 Enter your problem (or 'exit' to quit): ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'q', 'bye']:
                print("\n👋 Goodbye! Thanks for using MetaMind.")
                break
            
            if user_input.lower() == 'history':
                # Get current state from checkpointer
                state = app.get_state(config)
                if state and state.values.get("messages"):
                    print("\n📜 Conversation History:")
                    for i, msg in enumerate(state.values["messages"]):
                        role = "User" if isinstance(msg, HumanMessage) else "AI"
                        content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                        print(f"  [{i+1}] {role}: {content}")
                else:
                    print("No history yet.")
                continue
            
            if user_input.lower() == 'reset':
                # Start a new thread
                thread_id = f"interactive_session_{id(user_input)}"
                config["configurable"]["thread_id"] = thread_id
                print("✅ History cleared. Starting fresh.")
                continue
            
            if not user_input:
                print("Please enter a problem description.")
                continue
            
            # Check if we have existing state
            current_state = app.get_state(config)
            
            if current_state and current_state.values.get("messages"):
                # Continue existing conversation - only update input and add new message
                input_state = {
                    "input": user_input,
                    "messages": [HumanMessage(content=user_input)],
                    # Reset these for new problem
                    "plan": None,
                    "current_step_index": 0,
                    "past_steps": [],
                    "final_response": None,
                    "execution_result": None,
                    "iteration_count": 0,
                    "should_replan": False,
                    # Keep the model store for continuity
                    "model_store": current_state.values.get("model_store", {})
                }
            else:
                # First message - create initial state
                input_state = create_initial_state(user_input)
            
            print("\n" + "─" * 50)
            print("🔄 Processing your request...")
            print("─" * 50)
            
            final_state = None
            for event in app.stream(input_state, config=config):
                for node_name, node_output in event.items():
                    if node_name == "__end__":
                        continue
                    
                    print(f"\n📍 [{node_name.upper()}]")
                    
                    if node_name == "planner":
                        if node_output.get("plan"):
                            plan = node_output["plan"]
                            print(f"   Problem type: {plan.problem_type}")
                            print(f"   Selected method: {plan.selected_method}")
                            print(f"   Confidence: {plan.confidence:.0%}")
                            print(f"   Steps: {len(plan.steps)}")
                    
                    elif node_name == "agent":
                        step_idx = node_output.get("current_step_index", 0)
                        past_steps = node_output.get("past_steps", [])
                        if past_steps:
                            last_step, last_result = past_steps[-1]
                            print(f"   Executed step {last_step.step_id}: {last_step.description[:50]}...")
                            if isinstance(last_result, dict):
                                status = last_result.get("status", "unknown")
                                print(f"   Status: {status}")
                                if "best_fitness" in last_result:
                                    print(f"   Best fitness: {last_result['best_fitness']}")
                                if "model_id" in last_result:
                                    print(f"   Model ID: {last_result['model_id']}")
                    
                    elif node_name == "replanner":
                        if node_output.get("final_response"):
                            print(f"   ✅ Task completed!")
                    
                    final_state = node_output
            
            print("\n" + "═" * 50)
            print("📊 FINAL RESULT")
            print("═" * 50)
            
            if final_state:
                if final_state.get("final_response"):
                    print(final_state["final_response"])
                elif final_state.get("past_steps"):
                    print("\nExecution Summary:")
                    for step, result in final_state["past_steps"]:
                        status = "✅" if isinstance(result, dict) and result.get("status") == "success" else "❌"
                        print(f"  {status} Step {step.step_id}: {step.description[:40]}...")
                        if isinstance(result, dict):
                            if "best_fitness" in result:
                                print(f"      Best fitness: {result['best_fitness']}")
                            if "best_tour" in result:
                                print(f"      Best tour: {result['best_tour']}")
                            if "predictions" in result:
                                print(f"      Predictions: {result['predictions']}")
                else:
                    print("No results available.")
            
            # Show message count for debugging
            state_after = app.get_state(config)
            if state_after and state_after.values.get("messages"):
                print(f"\n📝 Total messages in history: {len(state_after.values['messages'])}")
            
            print("═" * 50)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            import traceback
            print(f"\n❌ Error: {e}")
            traceback.print_exc()
            print("Please try again with a different input.")
            continue


def main():
    graph = build_graph()
    
    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)
    
    print("Graph structure:")
    visualize_graph(app)
    print()
    
    run_interactive(app)