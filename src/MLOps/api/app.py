from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
import sys
import traceback
import logging
from pathlib import Path

# Simple path setup
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import GeminiRAGSystem - FROM YOUR STRUCTURE
try:
    # Your gemini_rag_system.py is at: src/model/gemini_rag_system.py
    from src.model.gemini_rag_system import GeminiRAGSystem
    logger.info("✅ Imported GeminiRAGSystem from src.model")
except ImportError as e:
    logger.error(f"❌ Import failed: {e}")
    # Fallback: manual import
    gemini_system_path = project_root / "src" / "model" / "gemini_rag_system.py"
    if gemini_system_path.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("gemini_rag_system", gemini_system_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        GeminiRAGSystem = module.GeminiRAGSystem
        logger.info("✅ Imported GeminiRAGSystem via manual import")
    else:
        logger.error(f"❌ File not found: {gemini_system_path}")
        raise

# Check if we're in cloud environment
def is_cloud_environment():
    """Detect if running on cloud platform"""
    return any([
        os.getenv('RAILWAY_ENVIRONMENT_NAME'),
        os.getenv('RENDER') == 'true',
        os.getenv('DYNO'),
        os.getenv('VERCEL'),
    ])

# Initialize FastAPI
app = FastAPI(
    title="Personalized RAG Chatbot API",
    description="AI Chatbot with RAG capabilities",
    version="1.0.0"
)

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    use_history: bool = True

class ChatResponse(BaseModel):
    success: bool
    answer: str
    sources_count: int
    response_time: float

@app.get("/")
async def root():
    return {"message": "Personalized RAG Chatbot API is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "chatbot-api"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Chat with the Gemini RAG chatbot
    """
    import time
    start_time = time.time()
    
    try:
        # Initialize chatbot once on first request
        if not hasattr(app, 'chatbot'):
            logger.info("Initializing Gemini RAG System...")
            use_small = is_cloud_environment()
            app.chatbot = GeminiRAGSystem(use_small_model=use_small)
            logger.info("✅ Gemini RAG System initialized!")
        
        # Process the request
        result = app.chatbot.ask_question(request.message, request.use_history)
        response_time = time.time() - start_time
        
        return ChatResponse(
            success=result['success'],
            answer=result['answer'],
            sources_count=result.get('sources_count', 0),
            response_time=response_time
        )
        
    except Exception as e:
        response_time = time.time() - start_time
        logger.error(f"Error in chat endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        
        return ChatResponse(
            success=False,
            answer=f"Error: {str(e)}",
            sources_count=0,
            response_time=response_time
        )

@app.get("/conversation/history")
async def get_conversation_history():
    """Get current conversation history"""
    try:
        if hasattr(app, 'chatbot'):
            return {
                "history": app.chatbot.conversation_history,
                "total_turns": len(app.chatbot.conversation_history)
            }
        else:
            return {"history": [], "total_turns": 0}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/conversation/clear")
async def clear_conversation_history():
    """Clear conversation history"""
    try:
        if hasattr(app, 'chatbot'):
            app.chatbot.conversation_history.clear()
            return {"message": "Conversation history cleared"}
        else:
            return {"message": "No chatbot instance found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def start_server():
    # ✅ Use the same forced port
    port = 8000
    host = "0.0.0.0"
    
    print(f"🔧 Starting server on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,  # ✅ Same port
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    start_server()