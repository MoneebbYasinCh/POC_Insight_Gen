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
        ("system", "You are an expert marketing analyst. Analyze the following campaign data and provide helpful, actionable insights for the user question."),
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