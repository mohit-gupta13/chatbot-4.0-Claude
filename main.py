from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Global state for FAQ data and Vectorizer
faq_cache = {
    "data": [],
    "matrix": None,
    "vectorizer": None,
    "mtime": 0
}

class ChatRequest(BaseModel):
    message: str

class FeedbackRequest(BaseModel):
    question: str
    helpful: bool

def log_unanswered_query(question: str, reason: str):
    print(f"DEBUG: Logging query: '{question}' for reason: '{reason}'")
    file_path = "unanswered_queries.json"
    entry = {
        "question": question,
        "reason": reason,
        "timestamp": datetime.now().isoformat()
    }
    
    current_data = []
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                current_data = json.load(f)
        except json.JSONDecodeError:
            print("DEBUG: Error decoding JSON, starting fresh.")
            pass
            
    current_data.append(entry)
    
    with open(file_path, "w") as f:
        json.dump(current_data, f, indent=4)
    print("DEBUG: Log saved successfully.")

def get_faq_data():
    file_path = "faq.json"
    try:
        current_mtime = os.path.getmtime(file_path)
    except FileNotFoundError:
        return [], None, None

    # Reload if file has changed
    if current_mtime > faq_cache["mtime"]:
        print(f"DEBUG: Reloading FAQ data (mtime: {current_mtime})")
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            
            if not data:
                return [], None, None

            # Re-initialize TF-IDF
            vec = TfidfVectorizer(stop_words='english')
            qs = [item["question"] for item in data]
            matrix = vec.fit_transform(qs)
            
            # Update cache
            faq_cache["data"] = data
            faq_cache["vectorizer"] = vec
            faq_cache["matrix"] = matrix
            faq_cache["mtime"] = current_mtime
        except Exception as e:
            print(f"DEBUG: Error loading FAQ: {e}")
            return faq_cache["data"], faq_cache["matrix"], faq_cache["vectorizer"]

    return faq_cache["data"], faq_cache["matrix"], faq_cache["vectorizer"]

def find_best_match(user_question):
    data, matrix, vec = get_faq_data()
    if not data or matrix is None or vec is None:
        return None

    # Transform user question
    user_tfidf = vec.transform([user_question])
    
    # Calculate cosine similarity
    similarities = cosine_similarity(user_tfidf, matrix)
    
    # Get the best match index and score
    best_idx = np.argmax(similarities)
    best_score = similarities[0, best_idx]
    
    print(f"DEBUG: Best score for '{user_question}': {best_score:.4f}")

    # Check strict threshold > 0.30
    if best_score > 0.30:
        return data[best_idx]
        
    return None

@app.post("/chat")
async def chat(request: ChatRequest):
    user_message = request.message
    best_match = find_best_match(user_message)
    
    if best_match:
        # Determine if feedback should be shown
        is_greeting = best_match["question"].lower() in ["hi", "hello", "hey"]
        return {
            "response": best_match["answer"],
            "show_feedback": not is_greeting
        }
    else:
        log_unanswered_query(user_message, "unanswered")
        return {
            "response": "I apologize, but I am unable to answer that. Please contact our customer support for assistance.",
            "show_feedback": False
        }

@app.post("/feedback")
async def feedback(request: FeedbackRequest):
    if not request.helpful:
        log_unanswered_query(request.question, "negative_feedback")
    return {"status": "received"}

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r") as f:
        return f.read()

def main():
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    main()
