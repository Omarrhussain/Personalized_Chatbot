import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple

# Add config to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

logger = logging.getLogger(__name__)

class GeminiRAGSystem:
    def __init__(self, vector_db_path: str = None, use_small_model: bool = None):
        """
        Initialize Gemini RAG System
        
        Args:
            vector_db_path: Custom path to vector database
            use_small_model: If True, use gemini-rag-small. If False, use gemini-rag.
        """
        # Determine which model to use
        if use_small_model is None:
            # Auto-detect: check if running on cloud
            is_cloud = any([
                os.getenv('RAILWAY_ENVIRONMENT_NAME'),
                os.getenv('RENDER') == 'true',
                os.getenv('DYNO'),
                os.getenv('VERCEL'),
            ])
            use_small_model = is_cloud
        
        self.use_small_model = use_small_model
        model_name = "gemini-rag-small" if use_small_model else "gemini-rag"
        logger.info(f"Using vector database: {model_name}")
        
        # Set vector database path
        if vector_db_path is None:
            possible_paths = [
                Path(__file__).parent.parent.parent / "model" / model_name,
                Path(__file__).parent.parent.parent / model_name,
                Path("/app/model") / model_name,
                Path("/app") / model_name,
                Path("/opt/render/project/src/model") / model_name,
                Path("/opt/render/project") / model_name,
            ]
            
            self.vector_db_path = None
            for path in possible_paths:
                if path.exists():
                    self.vector_db_path = path
                    logger.info(f"Found vector database at: {path}")
                    break
            
            if not self.vector_db_path:
                self.vector_db_path = possible_paths[0]
                logger.warning(f"Vector DB not found, using: {self.vector_db_path}")
        else:
            self.vector_db_path = Path(vector_db_path)
        
        # Load vector database
        self.vector_db = self._load_vector_db()
        self.retriever = self.vector_db.as_retriever(search_kwargs={"k": 3})
        self.model = self._initialize_gemini()
        self.conversation_history: List[Tuple[str, str]] = []
        
        logger.info("Gemini RAG System initialized successfully!")

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

    def _initialize_gemini(self):
        """Initialize Gemini model using config/api_keys.py"""
        try:
            # Import API key from your config file
            try:
                from config.api_keys import GEMINI_API_KEY
                api_key = GEMINI_API_KEY
                logger.info("✅ Loaded Gemini API key from config/api_keys.py")
            except ImportError:
                # Fallback to environment variable
                api_key = os.getenv('GEMINI_API_KEY')
                if not api_key:
                    raise ValueError("GEMINI_API_KEY not found in config/api_keys.py or environment variables")
                logger.info("✅ Loaded Gemini API key from environment variable")
            
            # Configure Gemini
            genai.configure(api_key=api_key)
            
            # Use gemini-2.5-flash model
            model = genai.GenerativeModel('gemini-2.5-flash')  # Updated to your model
            logger.info("✅ Gemini 2.5 Flash model initialized!")
            return model
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini model: {str(e)}")
            raise

    def ask_question(self, question: str, use_history: bool = True) -> Dict:
        """Ask question with RAG context"""
        try:
            # Get relevant context
            docs = self.retriever.invoke(question)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Build prompt with history
            history_text = ""
            if use_history and self.conversation_history:
                history_text = "\nPrevious conversation:\n"
                for q, a in self.conversation_history[-3:]:  # Last 3 exchanges
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
                if len(self.conversation_history) > 5:  # Keep last 5 exchanges
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

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()
        logger.info("🗑️ Conversation history cleared")