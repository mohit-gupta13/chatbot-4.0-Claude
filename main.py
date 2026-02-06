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

# Mount static files if directory exists
if os.path.exists("static"):
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

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    try:
        print("=" * 60)
        print("🚀 Initializing Travel Agent AI Chatbot")
        print("=" * 60)
        
        # Validate configuration
        Config.validate()
        print("✓ Configuration validated")
        
        # Initialize Vector DB Knowledge Base
        vector_service = get_vector_service()
        vector_service.initialize_knowledge_base()
        print("✓ Vector database initialized")
        
        # Warm up LLM service (initializes AWS Bedrock connection)
        llm_service = get_llm_service()
        print("✓ AWS Bedrock Claude connection established")
        
        # Test data service
        data_service = get_data_service()
        columns = data_service.get_column_names()
        print(f"✓ Data service loaded ({len(columns)} columns found)")
        
        print("=" * 60)
        print("✅ System initialized successfully")
        print("=" * 60)
        
    except Exception as e:
        print("=" * 60)
        print(f"❌ Startup Error: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Main GenAI chat endpoint.
    Uses AWS Bedrock Claude with LangChain Agent and RAG.
    """
    try:
        user_message = request.message
        
        # Initialize LLM service
        llm_service = get_llm_service()
        
        # Process query with LangChain Agent
        # The agent handles tool calling internally and returns natural language response
        llm_result = llm_service.process_query(user_message)
        
        if not llm_result.get("success"):
            return {
                "response": llm_result.get("error", "Failed to process query"),
                "response_type": "error"
            }
        
        # Return agent's response
        return {
            "response": llm_result["response"],
            "response_type": "text"
        }
                
    except Exception as e:
        print(f"CHAT ERROR: {e}")
        import traceback
        traceback.print_exc()
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
        "llm_provider": "AWS Bedrock",
        "llm_model": Config.BEDROCK_MODEL_ID,
        "vector_db": "Chroma",
        "embedding_model": Config.EMBEDDING_MODEL,
        "data_file": Config.BOOKING_FILE,
        "data_found": os.path.exists(Config.BOOKING_FILE),
        "chroma_dir": Config.CHROMA_DB_DIR,
        "chroma_exists": os.path.exists(Config.CHROMA_DB_DIR)
    }

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    html_path = "static/index.html"
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return """
        <html>
            <head><title>Travel Agent Chatbot</title></head>
            <body>
                <h1>Travel Agent AI Chatbot</h1>
                <p>API is running. Access the chat interface at <code>/static/index.html</code></p>
                <p>Check system status at <code>/status</code></p>
            </body>
        </html>
        """

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
