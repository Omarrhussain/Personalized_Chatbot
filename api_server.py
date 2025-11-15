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

# Add src to Python path
current_dir = os.path.dirname(__file__)
project_root = current_dir
src_path = os.path.join(project_root, 'src')

for path in [project_root, src_path]:
    if path not in sys.path:
        sys.path.insert(0, path)

if __name__ == "__main__":
    # ✅ FORCE port 8000 (ignore any other port)
    port = 8000
    host = "0.0.0.0"
    
    print(f"🚀 Starting on {host}:{port}")
    print(f"📍 Forced port: {port}")
    
    uvicorn.run(
        "src.MLOps.api.app:app",
        host=host,
        port=port,  # ✅ Now it will always use 8000
        reload=False,
        log_level="info"
    )