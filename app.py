import streamlit as st
from input_layer import is_query_needed, llm_direct_response
from db_tool import fetch_campaigns
from insight_gen import analyze_campaign_data

st.title("Prompt Query Decision & LLM Response")

# Initialize chat history in session state
if "history" not in st.session_state:
    st.session_state.history = []  # List of (role, content) tuples

user_prompt = st.text_input("Enter your prompt:")
limit = st.number_input("Max results to display", min_value=1, max_value=100, value=5)

if user_prompt:
    # Add user message to history
    st.session_state.history.append(("user", user_prompt))
    result = is_query_needed(user_prompt, history=st.session_state.history[:-1])  # Exclude current user input
    st.subheader("Decision:")
    st.json(result.dict())
    if result.query_needed:
        query, db_results = fetch_campaigns(user_prompt, limit=limit)
        st.subheader("MongoDB Query Used:")
        st.code(str(query), language="python")
        st.subheader("Database Results:")
        if db_results:
            st.json(db_results)
            st.subheader("Insights:")
            insights = analyze_campaign_data(db_results, user_prompt)
            st.write(insights)
            # Add assistant response to history
            st.session_state.history.append(("assistant", f"DB Results: {db_results}\nInsights: {insights}"))
        else:
            st.info("No matching campaigns found.")
            st.session_state.history.append(("assistant", "No matching campaigns found."))
    else:
        llm_response = llm_direct_response(user_prompt)
        st.subheader("LLM Response:")
        st.write(llm_response)
        st.session_state.history.append(("assistant", llm_response)) 