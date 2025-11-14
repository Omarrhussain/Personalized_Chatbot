from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
import sys
import traceback
import logging
from pathlib import Path

# Fix path for Railway deployment
def get_vector_db_path():
    # Try multiple possible paths
    possible_paths = [
        Path("model/gemini-rag-small"),  # Local development
        Path("/app/model/gemini-rag-small"),  # Railway default
        Path("./model/gemini-rag-small"),  # Relative path
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"✅ Found vector database at: {path}")
            return str(path)
    
    # List what directories actually exist
    print("🔍 Searching for model directory...")
    for root, dirs, files in os.walk("/app"):
        for dir in dirs:
            if "model" in dir.lower():
                print(f"Found directory: {os.path.join(root, dir)}")
    
    raise FileNotFoundError("Vector database not found in any expected location")

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Gemini RAG Chatbot API",
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
    return {"message": "Gemini RAG Chatbot API is running!"}

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
        # Import here to catch import errors
        from src.model.gemini_rag_system import GeminiRAGSystem
        
        # Initialize chatbot (do it here to catch init errors)
        if not hasattr(app, 'chatbot'):
            logger.info("Initializing Gemini RAG System...")
            app.chatbot = GeminiRAGSystem()
            logger.info("✅ Gemini RAG System initialized successfully!")
        
        # Process the request
        result = app.chatbot.ask_question(request.message, request.use_history)
        response_time = time.time() - start_time
        
        logger.info(f"Chat request processed in {response_time:.2f}s")
        
        return ChatResponse(
            success=result['success'],
            answer=result['answer'],
            sources_count=result.get('sources_count', 0),
            response_time=response_time
        )
        
    except Exception as e:
        response_time = time.time() - start_time
        error_traceback = traceback.format_exc()
        
        logger.error(f"❌ Error in chat endpoint: {str(e)}")
        logger.error(f"Traceback: {error_traceback}")
        
        # Return detailed error for debugging
        return ChatResponse(
            success=False,
            answer=f"Server Error: {str(e)}",
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
    """Start the server with better configuration"""
    print("🚀 Starting Gemini RAG Chatbot API...")
    print("📍 Access URLs:")
    print("   • API Documentation: http://127.0.0.1:8000/docs")
    print("   • Health Check: http://127.0.0.1:8000/health") 
    print("⏹️  Press CTRL+C to stop the server")
    print("-" * 50)
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    start_server()