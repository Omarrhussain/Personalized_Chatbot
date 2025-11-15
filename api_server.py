#!/usr/bin/env python3
import uvicorn
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    uvicorn.run(
        "src.MLOps.api.app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )