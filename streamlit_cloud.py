import streamlit as st
import requests

st.set_page_config(page_title="Gemini Chatbot", page_icon="🤖")
st.title("🤖 Gemini RAG Chatbot")

# Your Railway URL - UPDATE AFTER DEPLOYMENT
API_URL = "https://personalized-chatbot-api-production.up.railway.app"  # ⬅️ CHANGE THIS

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat
for message in st.session_state.messages:
    if message["role"] == "user":
        st.write(f"**You:** {message['content']}")
    else:
        st.write(f"**Assistant:** {message['content']}")

# Chat input
user_input = st.text_input("Your message:")

if st.button("Send") and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.rerun()
    
    with st.spinner("Thinking..."):
        try:
            response = requests.post(
                f"{API_URL}/chat", 
                json={"message": user_input},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if data["success"]:
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": data["answer"]
                    })
            else:
                st.error("API connection failed")
        except:
            st.error("Cannot connect to API")
    
    st.rerun()

if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()