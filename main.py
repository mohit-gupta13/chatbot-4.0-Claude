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
    # You can expand this to save to a file or database if needed

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Main GenAI chat endpoint.
    Uses LLM function calling to query the booking database.
    """
    try:
        user_message = request.message
        
        # 1. Initialize services
        if not Config.is_configured():
            return {
                "response": "OpenAI API key is missing. Please configure it in the .env file.",
                "response_type": "error"
            }
        
        llm_service = get_llm_service()
        data_service = get_data_service()
        
        # 2. Process query with LLM
        llm_result = llm_service.process_query(user_message)
        
        if not llm_result.get("success"):
            return {
                "response": llm_result.get("error", "Failed to process query"),
                "response_type": "error"
            }
        
        # 3. Handle LLM Output
        if llm_result["type"] == "text_response":
            return {
                "response": llm_result["response"],
                "response_type": "text"
            }
        
        elif llm_result["type"] == "function_call":
            function_name = llm_result["function_name"]
            function_args = llm_result["function_args"]
            
            # Execute data fetch
            if function_name == "get_bookings":
                data_result = data_service.get_bookings(**function_args)
            else:
                return {
                    "response": f"Unsupported function: {function_name}",
                    "response_type": "error"
                }
            
            # Format and return result
            if data_result["success"]:
                summary = llm_service.format_data_response(
                    function_name, function_args, data_result
                )
                return {
                    "response": summary,
                    "response_type": "data",
                    "data": data_result.get("data", []),
                    "count": data_result.get("count", 0),
                    "function_called": function_name,
                    "function_args": function_args
                }
            else:
                return {
                    "response": data_result.get("error", "Data fetch failed"),
                    "response_type": "error"
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
        "api_configured": Config.is_configured(),
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
