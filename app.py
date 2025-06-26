import streamlit as st
from input_layer import is_query_needed, llm_direct_response
from db_tool import fetch_campaigns
from insight_gen import analyze_campaign_data

st.set_page_config(page_title="AigentZ Chatbot", page_icon="🤖", layout="wide")
st.title("AigentZ Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

# Debug section - show history length and clear button
col1, col2 = st.columns([3, 1])
with col1:
    st.caption(f"Conversation history: {len(st.session_state.history)} messages")
with col2:
    if st.button("🗑️ Clear Chat"):
        st.session_state.history = []
        st.rerun()

limit = st.number_input("Max results to display", min_value=1, max_value=100, value=5)

# --- Chat display ---
st.markdown(
    """
    <style>
    .user-bubble {
        background-color: #DCF8C6;
        color: #222;
        padding: 10px 16px;
        border-radius: 18px 18px 2px 18px;
        margin-left: 40%;
        margin-bottom: 8px;
        text-align: right;
        max-width: 60%;
        float: right;
        clear: both;
    }
    .assistant-bubble {
        background-color: #F1F0F0;
        color: #222;
        padding: 10px 16px;
        border-radius: 18px 18px 18px 2px;
        margin-right: 40%;
        margin-bottom: 8px;
        text-align: left;
        max-width: 60%;
        float: left;
        clear: both;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Display chat history as bubbles
for role, content in st.session_state.history:
    if role == "user":
        st.markdown(f'<div class="user-bubble">{content}</div>', unsafe_allow_html=True)
    elif role == "assistant":
        st.markdown(f'<div class="assistant-bubble">{content}</div>', unsafe_allow_html=True)

# --- Chat input at the bottom ---
with st.form("chat_form", clear_on_submit=True):
    user_prompt = st.text_input("Enter your prompt:", key="chat_input")
    submitted = st.form_submit_button("Send")

if submitted and user_prompt.strip():
    st.session_state.history.append(("user", user_prompt))
    
    # Create containers for real-time updates
    status_container = st.empty()
    response_container = st.empty()
    
    assistant_response = ""
    
    # Step 1: Decision
    status_container.info("🤔 Thinking... Deciding if a DB query is needed.")
    result = is_query_needed(user_prompt, history=st.session_state.history[:-1])
    assistant_response += f"<b>Decision:</b> {result.model_dump()['reason']}<br>"
    response_container.markdown(f'<div class="assistant-bubble">{assistant_response}</div>', unsafe_allow_html=True)

    if result.query_needed:
        # Step 2: Query
        status_container.info("🔍 Querying the database...")
        query, db_results = fetch_campaigns(user_prompt, limit=limit)
        assistant_response += f"<b>MongoDB Query Used:</b> <code>{str(query)}</code><br>"
        response_container.markdown(f'<div class="assistant-bubble">{assistant_response}</div>', unsafe_allow_html=True)
        
        # Step 3: Results
        status_container.info("📊 Fetching database results...")
        if db_results:
            assistant_response += "<b>Database Results:</b><br>" + f"<pre>{db_results}</pre>"
            response_container.markdown(f'<div class="assistant-bubble">{assistant_response}</div>', unsafe_allow_html=True)
            
            # Step 4: Insights
            status_container.info("💡 Generating insights...")
            insights = analyze_campaign_data(db_results, user_prompt)
            assistant_response += f"<b>Insights:</b> {insights}"
            response_container.markdown(f'<div class="assistant-bubble">{assistant_response}</div>', unsafe_allow_html=True)
        else:
            assistant_response += "No matching campaigns found."
            response_container.markdown(f'<div class="assistant-bubble">{assistant_response}</div>', unsafe_allow_html=True)
    else:
        with st.spinner("Generating LLM response..."):
            llm_response = llm_direct_response(user_prompt, history=st.session_state.history[:-1])
            llm_response_text = f"<b>LLM Response:</b> {llm_response}"
            response_container.markdown(f'<div class="assistant-bubble">{llm_response_text}</div>', unsafe_allow_html=True)
    
    # Clear status and add final response to history
    status_container.empty()
    st.session_state.history.append(("assistant", assistant_response)) 