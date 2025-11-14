import streamlit as st
import requests

# Page configuration
st.set_page_config(
    page_title="Personalized RAG Chatbot - Cloud",
    page_icon="🤖",
    layout="wide"
)

st.markdown('<div style="font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 2rem;">🤖 Personalized RAG Chatbot - Cloud</div>', unsafe_allow_html=True)

# Your Railway URL - UPDATE THIS
RAILWAY_URL = "https://personalized-chatbot-api-production.up.railway.app/"  # ⬅️ CHANGE TO YOUR URL

with st.sidebar:
    st.header("🔧 Configuration")
    st.info(f"**API URL:** {RAILWAY_URL}")
    
    # Debug section
    st.markdown("---")
    st.header("🐛 Debug Tools")
    
    if st.button("Test /health Endpoint"):
        try:
            response = requests.get(f"{RAILWAY_URL}/health", timeout=10)
            st.write(f"Status: {response.status_code}")
            st.write(f"Response: {response.json()}")
        except Exception as e:
            st.error(f"Error: {e}")
    
    if st.button("Test /chat Endpoint"):
        try:
            response = requests.post(
                f"{RAILWAY_URL}/chat",
                json={"message": "Hello, are you working?", "use_history": False},
                timeout=30
            )
            st.write(f"Status: {response.status_code}")
            st.write(f"Full Response: {response.json()}")
        except Exception as e:
            st.error(f"Error: {e}")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
st.markdown("### 💬 Conversation")

for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**Assistant:** {message['content']}")
    st.markdown("---")

if not st.session_state.messages:
    st.info("💡 Start a conversation by typing a message below!")

# Chat input
user_input = st.text_input("Your message:", placeholder="Ask me anything...")

if st.button("Send Message") and user_input:
    # Add user message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.rerun()
    
    # Get bot response
    try:
        with st.spinner("🤔 Thinking..."):
            response = requests.post(
                f"{RAILWAY_URL}/chat",
                json={"message": user_input, "use_history": True},
                timeout=30
            )
            
        # Debug info
        st.sidebar.write(f"Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            st.sidebar.write(f"Success: {data.get('success', False)}")
            
            if data.get("success"):
                # Add bot response
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": data["answer"]
                })
                st.rerun()
            else:
                st.error(f"API Error: {data.get('answer', 'Unknown error')}")
        else:
            st.error(f"HTTP Error: {response.status_code}")
            
    except Exception as e:
        st.error(f"Connection error: {str(e)}")

if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()