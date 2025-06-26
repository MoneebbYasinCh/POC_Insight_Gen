from pymongo import MongoClient
from typing import Any, Dict, List, Tuple
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import ast

# MongoDB connection setup (edit URI as needed)
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "ecommerce"
COLLECTION_NAME = "campaigns"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# LLM agent for query generation (same as orchestrator/insight_gen)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    timeout=None,
    max_retries=2
)

def llm_generate_mongo_query(user_input: str) -> Dict:
    """
    Use LLM to generate a MongoDB query dict from user input. Returns dict if valid, else empty dict.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         '''You are an expert MongoDB query generator for a marketing campaigns database.
Your job is to convert user questions into valid MongoDB query dictionaries (Python dicts) for the 'campaigns' collection.
The schema for each campaign is:
{{{{
  "campaign_id": "001",
  "name": "Campaign_1",
  "channel": "Google",
  "region": "South Asia",
  "objective": "Video Views",
  "status": "Active",
  "date": "2025-06-25",
  "spend": 919.97,
  "impressions": 36795,
  "clicks": 1806,
  "ctr": 4.91,
  "conversions": 301
}}}}

Instructions:
- Always output ONLY a valid Python dictionary for the MongoDB query, and nothing else.
- If the user request is vague or general (e.g., 'show all campaigns', 'how are campaigns going?'), return {{{{}}}}.
- If the user asks for a filter, map it to the correct field(s) in the schema.
- For date ranges, use {{{{"date": {{{{"$gte": "YYYY-MM-DD", "$lte": "YYYY-MM-DD"}}}}}}}}.
- For numeric thresholds, use operators like {{{{"clicks": {{{{"$gt": 1000}}}}}}}}.
- If the user asks for a field that does not exist, return {{{{}}}}.
- Do not include sort, limit, or projection in the query.
- Do not explain or add any text, only output the dictionary.

Examples:
User: Show all active Google campaigns in South Asia
Output: {{{{"status": "Active", "channel": "Google", "region": "South Asia"}}}}

User: Campaigns with more than 1000 clicks
Output: {{{{"clicks": {{{{"$gt": 1000}}}}}}}}

User: Campaigns between 2025-06-01 and 2025-06-30
Output: {{{{"date": {{{{"$gte": "2025-06-01", "$lte": "2025-06-30"}}}}}}}}

User: How are campaigns going right now?
Output: {{{{}}}}

User: Show campaigns with objective "Video Views" and status "Active"
Output: {{{{"objective": "Video Views", "status": "Active"}}}}
'''),
        ("human", "User question: {user_input}")
    ])
    chain = prompt | llm
    result = chain.invoke({"user_input": user_input})
    query_str = result.content if hasattr(result, "content") else str(result)
    if not isinstance(query_str, str):
        query_str = str(query_str)
    try:
        # Safely evaluate the string to a Python dict
        query = ast.literal_eval(query_str)
        if isinstance(query, dict):
            return query
    except Exception:
        pass
    return {}

def generate_mongo_query(user_input: str) -> Dict:
    """
    Hybrid: Try LLM first, fallback to keyword mapping.
    """
    query = llm_generate_mongo_query(user_input)
    if query is not None:
        return query
    # Fallback: basic keyword mapping
    user_input = user_input.lower()
    if "active" in user_input:
        return {"status": "Active"}
    if "google" in user_input:
        return {"channel": "Google"}
    if "south asia" in user_input:
        return {"region": "South Asia"}
    return {}

def fetch_campaigns(user_input: str, limit: int = 5) -> Tuple[Dict, List[Dict[str, Any]]]:
    """
    Generate a MongoDB query from user input (hybrid LLM+mapping), fetch results, and return both.
    Also print the generated query to the terminal.
    """
    query = generate_mongo_query(user_input)
    print(f"[DB_TOOL] MongoDB Query Generated: {query}")
    results = list(collection.find(query).limit(limit))
    for doc in results:
        doc.pop("_id", None)
    return query, results 