from gemini_rag_system import GeminiRAGSystem

def main():
    chatbot = GeminiRAGSystem()
    print("🤖 Gemini RAG Chatbot Ready! Type 'quit' to exit.")
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == 'quit':
            break
            
        result = chatbot.ask_question(user_input)
        print(f"🤖: {result['answer']}")

if __name__ == "__main__":
    main()