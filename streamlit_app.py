import streamlit as st
import requests
import os

# Page configuration
st.set_page_config(
    page_title="Personalized RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Simple CSS for better readability (no colors)
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🤖 Personalized RAG Chatbot</div>', unsafe_allow_html=True)

# API Configuration
with st.sidebar:
    st.header("🔧 Configuration")
    
    api_url = st.text_input(
        "API Server URL", 
        value="http://127.0.0.1:8000",
        help="Make sure FastAPI is running on this URL"
    )
    
    st.markdown("---")
    st.markdown("### 🚀 Quick Start Guide")
    st.markdown("""
    1. **Open Terminal 1:**
    ```bash
    python api_server.py
    ```
    
    2. **Open Terminal 2:**
    ```bash
    streamlit run streamlit_app.py
    ```
    """)

# Function to check API health
def check_api_health(api_url):
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        return response.status_code == 200, "✅ API is running!"
    except requests.exceptions.ConnectionError:
        return False, "❌ API server is not running. Start it with: `python api_server.py`"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"

# Health check on startup
health_status, health_message = check_api_health(api_url)
if health_status:
    st.sidebar.success(health_message)
else:
    st.sidebar.error(health_message)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages - SIMPLE VERSION WITHOUT COLORS
st.markdown("### 💬 Conversation")

# Show all messages in the chat
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**Assistant:** {message['content']}")
        if message.get("sources"):
            st.caption(f"📚 Used {message['sources']} knowledge sources")
    st.markdown("---")

# Show empty state if no messages
if not st.session_state.messages:
    st.info("💡 Start a conversation by typing a message below!")

# Chat input
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Your message:", placeholder="Ask me anything about AI, machine learning, etc...", key="chat_input")
    submitted = st.form_submit_button("🚀 Send Message")

if submitted and user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Get bot response
    try:
        with st.spinner("🤔 Thinking..."):
            response = requests.post(
                f"{api_url}/chat",
                json={"message": user_input, "use_history": True},
                timeout=60
            )
            
        if response.status_code == 200:
            data = response.json()
            if data["success"]:
                # Add bot response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": data["answer"],
                    "sources": data.get("sources_count", 0)
                })
                st.success("✅ Response received!")
                st.rerun()  # Refresh to show new messages
            else:
                st.error(f"❌ API Error: {data['answer']}")
                st.rerun()
                
        else:
            st.error(f"❌ HTTP Error: {response.status_code}")
            st.rerun()
            
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Connection error: {str(e)}")
        st.rerun()

# Clear chat button
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

# Manual health check button
if st.sidebar.button("🩺 Check API Health"):
    health_status, health_message = check_api_health(api_url)
    if health_status:
        st.sidebar.success(health_message)
    else:
        st.sidebar.error(health_message)