from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from typing import Any, List, Dict

# Define a separate LLM agent for insights
insight_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    timeout=None,
    max_retries=2
)

def analyze_campaign_data(data: List[Dict[str, Any]], user_input: str) -> Any:
    """
    Pass the fetched campaign data and user input to the insight agent for analysis and feedback.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant that can analyze data and answer various types of questions.

IMPORTANT: Focus on answering the user's actual question, not just providing campaign insights.

- If the user asks about the conversation history or previous prompts, answer that directly.
- If the user asks about the data you received, analyze and explain it.
- If the user asks for insights about campaigns, provide marketing analysis.
- If the user asks general questions, answer them appropriately.
- Always respond to what the user is actually asking, not what you think they should be asking."""),
        ("human", "User question: {user_input}\n\nCampaign data: {campaign_data}")
    ])
    chain = prompt | insight_llm
    # Format the data as a string for the prompt
    data_str = str(data)
    result = chain.invoke({"user_input": user_input, "campaign_data": data_str})
    if hasattr(result, "content"):
        return result.content
    if isinstance(result, list):
        return "\n".join(str(item) for item in result)
    if isinstance(result, dict):
        return str(result)
    return str(result) 