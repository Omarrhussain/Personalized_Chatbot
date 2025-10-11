import os
import pandas as pd
from src.model.inference import FastChatbot

class ChatbotPipeline:
    def __init__(self, model_path="models/fast-chatbot"):
        self.chatbot = FastChatbot(model_path)
        print("✅ Chatbot Pipeline Ready!")
    
    def interactive_chat(self):
        """Start interactive chat session"""
        print("🤖 Chatbot Activated! Type 'quit' to exit.")
        print("-" * 50)
        
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Bot: Goodbye! 👋")
                break
            
            response = self.chatbot.chat(user_input)
            print(f"Bot: {response}")
    
    def batch_test(self, test_file="data/test_samples.txt"):
        """Test chatbot with sample inputs"""
        if os.path.exists(test_file):
            with open(test_file, 'r') as f:
                samples = [line.strip() for line in f if line.strip()]
            
            print("🧪 Batch Testing:")
            for sample in samples:
                response = self.chatbot.chat(sample)
                print(f"Input: {sample}")
                print(f"Response: {response}")
                print("-" * 40)

if __name__ == "__main__":
    pipeline = ChatbotPipeline()
    pipeline.interactive_chat()