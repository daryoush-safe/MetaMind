from dotenv import load_dotenv

from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from IPython.display import Image, display

from my_agent.agent import build_graph
from my_agent.utils.state import create_initial_state


# =========================================================================
# Helpers
# =========================================================================

def visualize_graph(app, path: str = "graph.png"):
    """Render the graph as a Mermaid diagram and save it as a PNG."""
    try:
        png_bytes = app.get_graph(xray=True).draw_mermaid_png()
        with open(path, "wb") as f:
            f.write(png_bytes)
        display(Image(png_bytes))
    except Exception as e:
        print(f"Could not visualize: {e}")


# =========================================================================
# Interactive REPL
# =========================================================================

def run_interactive(app):
    """Run MetaMind in an interactive terminal loop with persistent memory."""
    print("=" * 60)
    print("  MetaMind: LLM-Orchestrated Computational Intelligence")
    print("=" * 60)
    print()
    print("Available problem types:")
    print("  - TSP / Routing problems  (ACO, GA)")
    print("  - Function optimization   (PSO)")
    print("  - Classification          (MLP, Perceptron)")
    print("  - Clustering              (SOM)")
    print("  - Pattern completion       (Hopfield)")
    print("  - Control / Regression     (Fuzzy)")
    print("  - Symbolic regression      (GP)")
    print()
    print("Commands:  'exit'    -- quit")
    print("           'history' -- show conversation history")
    print("           'reset'   -- clear history and start fresh")
    print("=" * 60)
    print()

    thread_id = "interactive_session"
    config = {
        "recursion_limit": 50,
        "configurable": {"thread_id": thread_id},
    }

    while True:
        try:
            user_input = input("\n🧠 Enter your problem (or 'exit' to quit): ").strip()

            # ── Meta-commands ────────────────────────────────────────
            if user_input.lower() in ("exit", "quit", "q", "bye"):
                print("\n👋 Goodbye! Thanks for using MetaMind.")
                break

            if user_input.lower() == "history":
                state = app.get_state(config)
                if state and state.values.get("messages"):
                    print("\n📜 Conversation History:")
                    for i, msg in enumerate(state.values["messages"]):
                        role = "User" if isinstance(msg, HumanMessage) else "AI"
                        content = (
                            msg.content[:100] + "..."
                            if len(msg.content) > 100
                            else msg.content
                        )
                        print(f"  [{i + 1}] {role}: {content}")
                else:
                    print("No history yet.")
                continue

            if user_input.lower() == "reset":
                thread_id = f"interactive_session_{id(user_input)}"
                config["configurable"]["thread_id"] = thread_id
                print("✅ History cleared. Starting fresh.")
                continue

            if not user_input:
                print("Please enter a problem description.")
                continue

            # ── Build input state ────────────────────────────────────
            current_state = app.get_state(config)

            if current_state and current_state.values.get("messages"):
                # Continue existing conversation
                input_state = {
                    "input": user_input,
                    "messages": [HumanMessage(content=user_input)],
                    # Reset per-problem fields
                    "plan": None,
                    "current_step_index": 0,
                    "past_steps": [],
                    "final_response": None,
                    "execution_result": None,
                    "iteration_count": 0,
                    "should_replan": False,
                    # Keep the model store for continuity
                    "model_store": current_state.values.get("model_store", {}),
                }
            else:
                input_state = create_initial_state(user_input)

            # ── Stream execution ─────────────────────────────────────
            print("\n" + "─" * 50)
            print("🔄 Processing your request...")
            print("─" * 50)

            final_state = None
            for event in app.stream(input_state, config=config):
                for node_name, node_output in event.items():
                    if node_name == "__end__":
                        continue

                    print(f"\n🔍 [{node_name.upper()}]")

                    if node_name == "planner":
                        plan = node_output.get("plan")
                        if plan:
                            print(f"   Problem type:     {plan.problem_type}")
                            print(f"   Selected method:  {plan.selected_method}")
                            print(f"   Confidence:       {plan.confidence:.0%}")
                            print(f"   Steps:            {len(plan.steps)}")

                    elif node_name == "agent":
                        past_steps = node_output.get("past_steps", [])
                        if past_steps:
                            last_step, last_result = past_steps[-1]
                            print(
                                f"   Executed step {last_step.step_id}: "
                                f"{last_step.description}"
                            )
                            if isinstance(last_result, dict):
                                status = last_result.get("status", "unknown")
                                print(f"   Status: {status}")
                                if "best_fitness" in last_result:
                                    print(
                                        f"   Best fitness: {last_result['best_fitness']}"
                                    )
                                if "model_id" in last_result:
                                    print(f"   Model ID: {last_result['model_id']}")

                    elif node_name == "replanner":
                        if node_output.get("final_response"):
                            print("   ✅ Task completed!")

                    final_state = node_output

            # ── Display final result ─────────────────────────────────
            print("\n" + "═" * 50)
            print("📊 FINAL RESULT")
            print("═" * 50)

            if final_state:
                if final_state.get("final_response"):
                    print(final_state["final_response"])
                elif final_state.get("past_steps"):
                    print("\nExecution Summary:")
                    for step, result in final_state["past_steps"]:
                        ok = isinstance(result, dict) and result.get("status") == "success"
                        icon = "✅" if ok else "❌"
                        print(f"  {icon} Step {step.step_id}: {step.description[:50]}...")
                        if isinstance(result, dict):
                            for key in ("best_fitness", "best_tour", "predictions"):
                                if key in result:
                                    print(f"      {key}: {result[key]}")
                else:
                    print("No results available.")

            state_after = app.get_state(config)
            if state_after and state_after.values.get("messages"):
                print(
                    f"\n📝 Total messages in history: "
                    f"{len(state_after.values['messages'])}"
                )

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


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    load_dotenv()

    graph = build_graph()

    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)

    print("Graph structure:")
    visualize_graph(app)
    print()

    run_interactive(app)