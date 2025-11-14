#!/usr/bin/env python3
"""
Simple starter for FastAPI server
"""
import uvicorn
import os
import sys

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    print("🚀 Starting Gemini RAG Chatbot API...")
    print("📍 Local Access URLs:")
    print("   • API: http://127.0.0.1:8000")
    print("   • Docs: http://127.0.0.1:8000/docs") 
    print("   • Health: http://127.0.0.1:8000/health")
    print("⏹️  Press CTRL+C to stop")
    print("-" * 50)
    
    uvicorn.run(
        "MLOps.api.app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )