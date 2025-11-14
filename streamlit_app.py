import streamlit as st
import requests

st.set_page_config(page_title="Personalized Chatbot", page_icon="🤖")
st.title("🤖 Personalized Chatbot")

# Use your Render URL after deployment
API_URL = "https://your-app-name.onrender.com"  # ← Change this after deployment

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display ALL messages (including new ones after rerun)
for message in st.session_state.messages:
    if message["role"] == "user":
        st.write(f"**You:** {message['content']}")
    else:
        st.write(f"**Assistant:** {message['content']}")

# Chat input
user_input = st.text_input("Your message:", placeholder="Ask me anything...")

if st.button("Send") and user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Get bot response
    with st.spinner("Thinking..."):
        try:
            response = requests.post(
                f"{API_URL}/chat",
                json={"message": user_input, "use_history": True},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if data["success"]:
                    # Add bot response to history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": data["answer"]
                    })
                    st.success("Response received!")
                    st.rerun()  # ← THIS refreshes to show new messages
                else:
                    st.error(f"Error: {data['answer']}")
            else:
                st.error(f"HTTP Error: {response.status_code}")
                
        except Exception as e:
            st.error(f"Connection error: {e}")

# Clear chat
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()