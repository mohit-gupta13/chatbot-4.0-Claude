from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from typing import Optional, Dict, Any, List

# Load configuration and services
from config import Config
from data_service import get_data_service
from llm_service import get_llm_service
from vector_service import get_vector_service

app = FastAPI(title="Travel Agent Back-Office Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

class ChatRequest(BaseModel):
    message: str

class FeedbackRequest(BaseModel):
    question: str
    helpful: bool

def log_feedback(question: str, helpful: bool):
    """Simple logging for user feedback"""
    print(f"FEEDBACK: Question: '{question}', Helpful: {helpful}")

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    try:
        # Initialize Vector DB Knowledge Base
        vector_service = get_vector_service()
        vector_service.initialize_knowledge_base()
        print("System initialized successfully.")
    except Exception as e:
        print(f"Startup Error: {e}")

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Main GenAI chat endpoint.
    Uses LangChain + Ollama Agent.
    """
    try:
        user_message = request.message
        
        # 1. Initialize services
        llm_service = get_llm_service()
        
        # 2. Process query with LLM Agent
        # The agent now handles the tool calling internally
        llm_result = llm_service.process_query(user_message)
        
        if not llm_result.get("success"):
            return {
                "response": llm_result.get("error", "Failed to process query"),
                "response_type": "error"
            }
        
        # 3. Return Response
        # LangChain agent returns the final natural language answer
        return {
            "response": llm_result["response"],
            "response_type": "text"
        }
                
    except Exception as e:
        print(f"CHAT ERROR: {e}")
        return {
            "response": f"An internal error occurred: {str(e)}",
            "response_type": "error"
        }

@app.post("/feedback")
async def feedback(request: FeedbackRequest):
    """Receive user feedback on chatbot performance"""
    log_feedback(request.question, request.helpful)
    return {"status": "received"}

@app.get("/status")
async def status():
    """Check system health"""
    return {
        "status": "online",
        "llm": Config.OLLAMA_MODEL,
        "vector_db": "chroma",
        "data_file": Config.BOOKING_FILE,
        "data_found": os.path.exists(Config.BOOKING_FILE)
    }

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r") as f:
        return f.read()

def main():
    import uvicorn
    # Validate config before starting
    try:
        Config.validate()
    except Exception as e:
        print(f"Warning: Configuration issue - {e}")
        
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    main()
