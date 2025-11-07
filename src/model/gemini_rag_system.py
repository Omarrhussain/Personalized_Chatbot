import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.api_keys import GEMINI_API_KEY

class GeminiRAGSystem:
    def __init__(self, vector_db_path: str = "models/gemini-rag"):
        self.vector_db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), vector_db_path)
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.vector_db = self._load_vector_db()
        self.retriever = self.vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        self.conversation_history = []
    
    def _load_vector_db(self) -> FAISS:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        return FAISS.load_local(self.vector_db_path, embeddings, allow_dangerous_deserialization=True)
    
    def ask_question(self, question: str) -> dict:
        try:
            docs = self.retriever.invoke(question)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Build prompt with history
            history_text = ""
            if self.conversation_history:
                history_text = "\nPrevious conversation:\n"
                for q, a in self.conversation_history[-3:]:
                    history_text += f"User: {q}\nAssistant: {a}\n"
            
            prompt = f"""Context: {context}{history_text}
Question: {question}
Answer:"""
            
            response = self.model.generate_content(prompt)
            self.conversation_history.append((question, response.text))
            
            return {
                'success': True,
                'answer': response.text,
                'sources_count': len(docs)
            }
        except Exception as e:
            return {'success': False, 'answer': f"Error: {str(e)}"}