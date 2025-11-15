import streamlit as st
import requests
import os

# Page configuration - EXACTLY LIKE streamlit_app.py
st.set_page_config(
    page_title="Personalized RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Simple CSS for better readability (no colors) - EXACTLY LIKE streamlit_app.py
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

# Header - EXACTLY LIKE streamlit_app.py
st.markdown('<div class="main-header">🤖 Personalized RAG Chatbot</div>', unsafe_allow_html=True)

# API Configuration - MODIFIED FOR CLOUD
with st.sidebar:
    st.header("🔧 Configuration")
    
    # Your Railway URL - UPDATE THIS with your actual Railway app URL
    RAILWAY_URL = os.getenv('RAILWAY_API_URL', "https://personalizedchatbot-production.up.railway.app")
    RAILWAY_URL = RAILWAY_URL.rstrip('/')  # Remove trailing slash
    
    # Allow override of URL for cloud
    api_url = st.text_input(
        "API Server URL", 
        value=RAILWAY_URL,
        help="Your Railway API URL"
    )
    
    st.markdown("---")
    st.markdown("### 🚀 Cloud Deployment")
    st.markdown("""
    This version connects to your **Railway API**.
    
    Make sure your API is deployed and running!
    """)

# Function to check API health - EXACTLY LIKE streamlit_app.py
def check_api_health(api_url):
    try:
        # INCREASE TIMEOUT from 5 to 30 seconds
        response = requests.get(f"{api_url}/health", timeout=30)
        return response.status_code == 200, "✅ API is running!"
    except requests.exceptions.ConnectionError:
        return False, "❌ API server is not running. Check your Railway deployment."
    except requests.exceptions.Timeout:
        return False, "❌ Connection timed out. Railway might be starting up."
    except Exception as e:
        return False, f"❌ Error: {str(e)}"

# Health check on startup - EXACTLY LIKE streamlit_app.py
health_status, health_message = check_api_health(api_url)
if health_status:
    st.sidebar.success(health_message)
else:
    st.sidebar.error(health_message)

# Initialize chat history - EXACTLY LIKE streamlit_app.py
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages - SIMPLE VERSION WITHOUT COLORS - EXACTLY LIKE streamlit_app.py
st.markdown("### 💬 Conversation")

# Show all messages in the chat - EXACTLY LIKE streamlit_app.py
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**Assistant:** {message['content']}")
        if message.get("sources"):
            st.caption(f"📚 Used {message['sources']} knowledge sources")
    st.markdown("---")

# Show empty state if no messages - EXACTLY LIKE streamlit_app.py
if not st.session_state.messages:
    st.info("💡 Start a conversation by typing a message below!")

# Chat input - EXACTLY LIKE streamlit_app.py
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Your message:", placeholder="Ask me anything about AI, machine learning, etc...", key="chat_input")
    submitted = st.form_submit_button("🚀 Send Message")

if submitted and user_input:
    # Add user message to chat history - EXACTLY LIKE streamlit_app.py
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Get bot response - MODIFIED FOR CLOUD ERROR HANDLING
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
                # Add bot response to chat history - EXACTLY LIKE streamlit_app.py
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

# Clear chat button - EXACTLY LIKE streamlit_app.py
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

# Manual health check button - EXACTLY LIKE streamlit_app.py
if st.sidebar.button("🩺 Check API Health"):
    health_status, health_message = check_api_health(api_url)
    if health_status:
        st.sidebar.success(health_message)
    else:
        st.sidebar.error(health_message)