from langgraph.graph import StateGraph, END
from .utils.state import AgentState, create_initial_state
from .utils.nodes import plan_step, execute_step, replan_step, should_continue
from IPython.display import Image, display

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

graph = build_graph()
app = graph.compile()

def visualize_graph():
    try:
        from IPython.display import Image, display
        display(Image(app.get_graph(xray=True).draw_mermaid_png()))
    except Exception as e:
        print(f"Could not visualize: {e}")

def run_interactive():
    """
    Run the MetaMind agent in interactive mode.
    
    User can keep entering problems, and the agent will solve them.
    Type 'exit' or 'quit' to stop.
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
    print("=" * 60)
    print()
    
    config = {
        "recursion_limit": 50,
        "configurable": {"thread_id": "interactive_session"}
    }
    
    while True:
        try:
            # Get user input
            user_input = input("\n🧠 Enter your problem (or 'exit' to quit): ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['exit', 'quit', 'q', 'bye']:
                print("\n👋 Goodbye! Thanks for using MetaMind.")
                break
            
            # Skip empty input
            if not user_input:
                print("Please enter a problem description.")
                continue
            
            # Create initial state
            initial_state = create_initial_state(user_input)
            
            print("\n" + "─" * 50)
            print("🔄 Processing your request...")
            print("─" * 50)
            
            # Stream the execution
            final_state = None
            for event in app.stream(initial_state, config=config):
                for node_name, node_output in event.items():
                    if node_name == "__end__":
                        continue
                    
                    print(f"\n📍 [{node_name.upper()}]")
                    
                    # Show relevant info based on node
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
                    
                    elif node_name == "re_planner":
                        if node_output.get("final_response"):
                            print(f"   ✅ Task completed!")
                    
                    final_state = node_output
            
            # Print final response
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
            
            print("═" * 50)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again with a different input.")
            continue


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    print("Graph structure:")
    visualize_graph()
    print()
    
    run_interactive()