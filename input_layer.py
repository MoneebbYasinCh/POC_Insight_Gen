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
escaped_format_instructions = format_instructions.replace("{", "{{{{").replace("}", "}}}}")

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

def escape_history_content(history):
    """Escape curly braces in history content to prevent template variable errors."""
    escaped_history = []
    for role, content in history:
        if isinstance(content, str):
            escaped_content = content.replace("{", "{{").replace("}", "}}")
            escaped_history.append((role, escaped_content))
        else:
            escaped_history.append((role, content))
    return escaped_history

# Add helper to build prompt messages with history
def build_prompt_messages(history, user_input, format_instructions):
    messages = [("system", """You are an assistant that decides if a user prompt needs a database query.

IMPORTANT GUIDELINES:
- If the user says a simple greeting (hi, hello, hey, good morning, etc.), respond directly without needing a database query.
- If the user asks about the conversation history (e.g., "what did we talk about?", "what was my last question?", "what was the previous prompt?"), respond directly without needing a database query.
- If the user asks meta-questions about the conversation itself, respond directly without needing a database query.
- Only suggest database queries for questions about marketing campaigns, data analysis, business insights, or specific data requests.
- Simple greetings, casual conversation, general questions, or conversation history questions should get friendly responses, not data queries.
- Focus on the CURRENT user input, not the conversation history, when making your decision.
- If the user asks for help, advice, or general information, respond directly without database queries.

EXAMPLES:
- "what was the last prompt i asked about?" → NO database query needed
- "what did we discuss earlier?" → NO database query needed  
- "show me active campaigns" → YES, database query needed
- "hi" → NO database query needed
- "how are campaigns performing?" → YES, database query needed""")]
    # Escape history content to prevent template variable errors
    escaped_history = escape_history_content(history)
    # Expand history
    for role, content in escaped_history:
        if role in ("user", "human"):
            messages.append(("human", content))
        elif role in ("assistant", "ai"):
            messages.append(("assistant", content))
    # Add current user input with format instructions appended
    messages.append(("human", f"{user_input}\n\n{format_instructions}"))
    return messages

def is_query_needed(user_input: str, history=None) -> QueryIntent:
    if history is None:
        history = []
    messages = build_prompt_messages(history, user_input, escaped_format_instructions)
    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | llm | parser
    return chain.invoke({"user_input": user_input})

# Function to get a direct LLM response if no query is needed
def llm_direct_response(user_input: str, history=None) -> Any:
    if history is None:
        history = []
    
    # Escape history content to prevent template variable errors
    escaped_history = escape_history_content(history)
    
    # Build messages list directly without template variables
    messages = [{"role": "system", "content": "You are a helpful ecommerce assistant and analyst named AigentZ."}]
    
    # Add conversation history
    for role, content in escaped_history:
        if role in ("user", "human"):
            messages.append({"role": "user", "content": content})
        elif role in ("assistant", "ai"):
            messages.append({"role": "assistant", "content": content})
    
    # Add current user input
    messages.append({"role": "user", "content": user_input})
    
    # Use the LLM directly with messages instead of a template
    result = llm.invoke(messages)
    
    # Handle different possible return types
    if hasattr(result, "content"):
        return result.content
    if isinstance(result, list):
        return "\n".join(str(item) for item in result)
    if isinstance(result, dict):
        return str(result)
    return str(result)

