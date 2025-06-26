from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
from typing import Any

load_dotenv()  # Loads from .env into environment variables

api_key = os.getenv("OPENAI_API_KEY")

# 1. Define schema
class QueryIntent(BaseModel):
    query_needed: bool = Field(..., description="True if database query is needed")
    reason: str = Field(..., description="Explanation of why a query is or isn't needed")

# 2. Parser
parser = PydanticOutputParser(pydantic_object=QueryIntent)

# 3. Render format instructions safely
format_instructions = parser.get_format_instructions()
escaped_format_instructions = format_instructions.replace("{", "{{").replace("}", "}}")

# 4. Prompt template (escaped)
# (Removed old static prompt template for is_query_needed)

# 5. LLM setup
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    timeout=None,
    max_retries=2
)
# (Removed old static chain for is_query_needed)

# Helper to format history for the prompt
def format_history(history):
    if not history:
        return ""
    return "\n".join([f"{role}: {content}" for role, content in history])

# Add helper to build prompt messages with history
def build_prompt_messages(history, user_input, format_instructions):
    messages = [("system", "You are an assistant that decides if a user prompt needs a database query.")]
    # Expand history
    for role, content in history:
        if role in ("user", "human"):
            messages.append(("human", content))
        elif role in ("assistant", "ai"):
            messages.append(("assistant", content))
    # Add current user input and format instructions
    messages.append(("human", user_input))
    messages.append(("assistant", format_instructions))
    return messages

def is_query_needed(user_input: str, history=None) -> QueryIntent:
    if history is None:
        history = []
    messages = build_prompt_messages(history, user_input, escaped_format_instructions)
    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | llm | parser
    return chain.invoke({"user_input": user_input})

# Function to get a direct LLM response if no query is needed
def llm_direct_response(user_input: str) -> Any:
    # Simple prompt for direct response
    direct_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful ecommerce assistant and analyst named AigentZ."),
        ("human", "{user_input}")
    ])
    direct_chain = direct_prompt | llm
    result = direct_chain.invoke({"user_input": user_input})
    # Handle different possible return types
    if hasattr(result, "content"):
        return result.content
    if isinstance(result, list):
        return "\n".join(str(item) for item in result)
    if isinstance(result, dict):
        return str(result)
    return str(result)

