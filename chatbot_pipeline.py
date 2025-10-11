import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model.inference import FastChatbot

def find_latest_checkpoint():
    """Find the latest checkpoint"""
    model_dir = "models/fine-tuned-chatbot"
    if not os.path.exists(model_dir):
        return None
    
    checkpoints = []
    for item in os.listdir(model_dir):
        if item.startswith("checkpoint-"):
            try:
                step = int(item.split("-")[1])
                checkpoints.append((step, os.path.join(model_dir, item)))
            except:
                continue
    
    if checkpoints:
        latest_step, latest_path = max(checkpoints, key=lambda x: x[0])
        print(f"✅ Using checkpoint: {latest_path} (step {latest_step})")
        return latest_path
    return None

class ChatbotPipeline:
    def __init__(self):
        print("🔄 Initializing Chatbot Pipeline...")
        
        # Find latest checkpoint
        adapter_path = find_latest_checkpoint()
        
        if adapter_path:
            self.chatbot = FastChatbot(adapter_path=adapter_path)
            self.model_type = "fine-tuned"
        else:
            # Fallback to base model only
            self.chatbot = FastChatbot(adapter_path=None)
            self.model_type = "base"
        
        print(f"✅ Chatbot Pipeline Ready ({self.model_type})")
    
    def chat(self, message):
        """Single message chat"""
        return self.chatbot.chat(message)
    
    def interactive_chat(self):
        """Interactive chat session"""
        print(f"🤖 Chatbot Started ({self.model_type})! Type 'quit' to exit.")
        print("-" * 40)
        
        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("Bot: Goodbye! 👋")
                    break
                
                response = self.chat(user_input)
                print(f"Bot: {response}")
            except KeyboardInterrupt:
                print("\nBot: Goodbye! 👋")
                break
            except Exception as e:
                print(f"Bot: Sorry, I encountered an error: {e}")

if __name__ == "__main__":
    try:
        pipeline = ChatbotPipeline()
        
        # Quick test
        print("🧪 Quick test:")
        response = pipeline.chat("Hello!")
        print(f"You: Hello!")
        print(f"Bot: {response}")
        
        # Interactive chat
        print("\n🎯 Starting interactive chat...")
        pipeline.interactive_chat()
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")