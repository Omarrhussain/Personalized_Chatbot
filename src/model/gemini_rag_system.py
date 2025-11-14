import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import sys
import logging
from pathlib import Path
from typing import Dict

# Add config to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

logger = logging.getLogger(__name__)

class GeminiRAGSystem:
   def __init__(self, vector_db_path: str = None):
    # Use absolute path to vector database
    if vector_db_path is None:
        # Try multiple deployment environments
        possible_paths = [
            Path(__file__).parent.parent.parent / "model" / "gemini-rag",  # Local
            Path("/app/model/gemini-rag-small"),  # Railway
            Path("/opt/render/project/src/model/gemini-rag-small"),  # Render
            Path("./model/gemini-rag-small"),  # Current directory
        ]
        
        for path in possible_paths:
            if path.exists():
                self.vector_db_path = path
                logger.info(f"✅ Found vector database at: {path}")
                break
        else:
            # If no path found, use local development path
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent
            self.vector_db_path = project_root / "model" / "gemini-rag"
    else:
        self.vector_db_path = Path(vector_db_path)
    
    logger.info(f"🔍 Using vector DB path: {self.vector_db_path}")
    
    def _load_vector_db(self) -> FAISS:
        """Load FAISS vector database"""
        logger.info(f"🔍 Looking for vector database at: {self.vector_db_path}")
        
        if not self.vector_db_path.exists():
            # List what's actually in the models directory
            models_dir = self.vector_db_path.parent
            if models_dir.exists():
                logger.info(f"📂 Contents of models directory: {list(models_dir.iterdir())}")
            raise FileNotFoundError(f"Vector database not found at {self.vector_db_path}")
        
        # Check for required files
        required_files = ['index.faiss', 'index.pkl']
        missing_files = [f for f in required_files if not (self.vector_db_path / f).exists()]
        
        if missing_files:
            raise FileNotFoundError(f"Missing vector database files: {missing_files}")
        
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Load with dangerous deserialization allowed
            vector_db = FAISS.load_local(
                str(self.vector_db_path), 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            logger.info("✅ Vector database loaded successfully!")
            return vector_db
        except Exception as e:
            raise Exception(f"Failed to load vector database: {str(e)}")
    
    def ask_question(self, question: str, use_history: bool = True) -> Dict:
        """Ask question with RAG context"""
        try:
            # Get relevant context - UPDATED API
            docs = self.retriever.invoke(question)  # ✅ Changed to invoke()
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Build prompt with history
            history_text = ""
            if use_history and self.conversation_history:
                history_text = "\nPrevious conversation:\n"
                for q, a in self.conversation_history[-3:]:
                    history_text += f"User: {q}\nAssistant: {a}\n"
            
            prompt = f"""Based on the following context, provide a helpful answer.

Context: {context}
{history_text}
Question: {question}

Please provide a clear and accurate response:"""

            # Generate response
            response = self.model.generate_content(prompt)
            
            # Update conversation history
            if use_history:
                self.conversation_history.append((question, response.text))
                if len(self.conversation_history) > 5:
                    self.conversation_history.pop(0)
            
            return {
                'success': True,
                'answer': response.text,
                'sources_count': len(docs)
            }
            
        except Exception as e:
            logger.error(f"Error in ask_question: {str(e)}")
            return {
                'success': False,
                'answer': f"Error: {str(e)}",
                'sources_count': 0
            }