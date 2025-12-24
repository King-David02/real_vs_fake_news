import openai
from atp_sdk.clients import LLMClient

openai_client = openai.OpenAI(api_key="YOUR_OPENAI_API_KEY")
llm_client = LLMClient(api_key="llm_atp_sk_xFsGJHVNM0QbzW916c5jx9jJFvg")

# Get toolkit context
context = llm_client.get_toolkit_context(
    toolkit_id="your_toolkit_id",
    provider="openai",
    user_prompt="Create a company and then list contacts."
)

# Use OpenAI to generate tool calls
response = openai_client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Create a company and then list contacts."}
    ],
    tools=context["tools"],
    tool_choice="auto"
)

# Extract and execute tool calls
tool_calls = response.choices[0].message.tool_calls

if tool_calls:

    result = llm_client.call_tool(
        toolkit_id="your_toolkit_id",
        tool_calls=tool_calls,
        provider="openai",
        user_prompt="Create a company and then list contacts."
    )

    print(f"Tool call result: {result}")