#!/usr/bin/env python3
"""
FastAPI Server for Personalized RAG Chatbot
Works both locally and on cloud (Railway, Render, etc.)
"""
import uvicorn
import os
import sys

# Add src to Python path - CORRECTED FOR YOUR STRUCTURE
current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(current_dir)  # Go up to project root
src_path = os.path.join(project_root, 'src')

if src_path not in sys.path:
    sys.path.insert(0, src_path)

def is_cloud_environment():
    """Detect if running on cloud platform"""
    return any([
        os.getenv('RAILWAY_ENVIRONMENT_NAME'),  # Railway
        os.getenv('RENDER') == 'true',  # Render
        os.getenv('DYNO'),  # Heroku
        os.getenv('VERCEL'),  # Vercel
    ])

if __name__ == "__main__":
    print("🚀 Starting Gemini RAG Chatbot API...")
    
    # Determine environment
    is_cloud = is_cloud_environment()
    port = int(os.getenv('PORT', 8080))
    host = "0.0.0.0" if is_cloud else "127.0.0.1"
    
    print(f"🌐 Environment: {'☁️ Cloud' if is_cloud else '💻 Local'}")
    print(f"📍 Server URL: {host}:{port}")
    print(f"📚 API Docs: http://{'localhost' if not is_cloud else host}:{port}/docs")
    print("⏹️  Press CTRL+C to stop")
    print("-" * 50)
    
    # CORRECT IMPORT PATH FOR YOUR STRUCTURE
    uvicorn.run(
        "src.MLOps.api.app:app",  # This is correct for your structure!
        host=host,
        port=port,
        reload=not is_cloud,  # Disable reload in cloud for stability
        log_level="info"
    )