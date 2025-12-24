from mistralai import Mistral
from atp_sdk.clients import LLMClient
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def print_turn(turn_type, content, prefix=""):
    """Helper function to print styled user or agent turns."""
    separator = "=" * 50
    print(f"\n{separator}\n{turn_type} TURN:\n{prefix}{content}\n{separator}")

# --- Role Instructions ---
TOOL_PLANNER_PROMPT = {
    "role": "system",
    "content": (
        "You are the Tool Planner.\n"
        "Your ONLY job is to decide which tools to call to satisfy the user's request.\n"
        "If a tool is needed, output a function call with correct parameters.\n"
        "Do NOT answer the question directly.\n"
        "If no tool is needed, simply respond with a short explanation and no tool call."
    )
}

FINAL_RESPONDER_PROMPT = {
    "role": "system",
    "content": (
        "You are the Final Responder.\n"
        "The tools have already been executed and their results are provided.\n"
        "Your job is to craft a clear, concise natural-language answer to the user.\n"
        "Do NOT call any tools.\n"
        "Summarize the tool results and address the original question."
    )
}

# --- Clients ---
mistral_client = Mistral(api_key="cwNX5ZUDYXdTdwZL5HYMKcvnw2kAKmfB")
llm_client = LLMClient(api_key="llm_atp_sk_xFsGJHVNM0QbzW916c5jx9jJFvg", protocol="https")

conversation_history = []  # Shared across turns


def main():
    toolkit_id = "serper_toolkit" # e.g., "serper_toolkit", "github_toolkit", "firecrawl_toolkit", "tavily_toolkit"
    auth_token = "ccc7c2d5596a22bc8375acdca7c503e2f6e25bd7" # e.g., Serper API key, GitHub token, Firecrawl API key, Tavily API key
    provider = "mistralai"

    print("Welcome to the Interactive ATP Terminal. Type your prompt or 'exit' to quit.")
    print("Example: 'Search for the Richest Man in the world.'")

    while True:
        try:
            # USER INPUT
            user_prompt = input("\nYour prompt: ").strip()
            if user_prompt.lower() == "exit":
                print_turn("SYSTEM", "Exiting interactive terminal.")
                break

            if not user_prompt:
                print_turn("SYSTEM", "Please enter a non-empty prompt.")
                continue

            print_turn("USER", user_prompt)

            # Fetch Toolkit Context
            print_turn("AGENT", "Fetching toolkit context...", prefix="> ")
            context = llm_client.get_toolkit_context(
                toolkit_id=toolkit_id,
                provider=provider,
                user_prompt=user_prompt
            )

            # Append user message to conversation
            conversation_history.append({"role": "user", "content": user_prompt})

            # --------- TOOL PLANNER STEP ---------
            print_turn("AGENT", "Generating tool plan...", prefix="> ")
            plan_response = mistral_client.chat.complete(
                model="mistral-large-2411",
                messages=[TOOL_PLANNER_PROMPT] + conversation_history,
                tools=context["tools"],
                tool_choice="auto"
            )

            tool_calls = plan_response.choices[0].message.tool_calls

            if tool_calls:

                # Append the assistant's tool-calling message to the conversation history
                conversation_history.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        # Reconstruct the tool call object
                        {"id": tc.id, "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in tool_calls
                    ]
                })

                # Show tool calls
                tool_call_summary = "\n".join(
                    f"Calling function '{tc.function.name}' with parameters: {tc.function.arguments}"
                    for tc in tool_calls
                )
                print_turn("AGENT", f"Executing tool calls:\n{tool_call_summary}", prefix="> ")

                # Execute tool calls
                results = llm_client.call_tool(
                    toolkit_id=toolkit_id,
                    tool_calls=tool_calls,
                    provider=provider,
                    user_prompt=user_prompt,
                    auth_token=auth_token
                )

                # Display tool execution results
                results_summary = "\n".join(
                    f"Tool '{r['name']}' (ID: {r['tool_call_id']}): {r['content'][:200]}..."  # Use 'content' (JSON string), truncate to avoid spam
                    for r in results
                )
                print_turn("AGENT", f"Tool execution results:\n{results_summary}", prefix="> ")
                
                for result in results:
                    conversation_history.append({
                        "role": result["role"],
                        "name": result["name"],
                        "content": result["content"],
                        "tool_call_id": result["tool_call_id"]
                    })

                # --------- FINAL RESPONDER STEP ---------
                print_turn("AGENT", "Generating final response...", prefix="> ")
                final_response = mistral_client.chat.complete(
                    model="mistral-large-2411",
                    messages=[FINAL_RESPONDER_PROMPT] + conversation_history,
                    # tools=[],          # No tools
                    # tool_choice="none" # Ensure no accidental tool calls
                )

                final_content = final_response.choices[0].message.content
                print_turn("AGENT", f"Final response:\n{final_content}", prefix="> ")

                # Append final assistant answer
                conversation_history.append({"role": "assistant", "content": final_content})

            else:
                # No tool call—use planner's natural response
                direct_content = plan_response.choices[0].message.content
                print_turn("AGENT", f"Direct response:\n{direct_content}", prefix="> ")
                conversation_history.append({"role": "assistant", "content": direct_content})

        except KeyboardInterrupt:
            print_turn("SYSTEM", "Exiting interactive terminal.")
            break
        except Exception as e:
            logger.error(f"Error in interactive loop: {e}")
            print_turn("SYSTEM", f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()