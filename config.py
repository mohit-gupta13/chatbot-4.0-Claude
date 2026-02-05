"""
Configuration module for GenAI Travel Chatbot
Manages API keys, file paths, and system settings
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration settings for the chatbot"""
    
    # Ollama Configuration
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
    
    # Vector DB Configuration
    CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
    
    # Data File Paths
    BOOKING_FILE = "flight Booking.xlsx"
    
    # System Settings
    TIMEOUT_SECONDS = 60
    
    # Logging
    LOG_LEVEL = "INFO"
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if not os.path.exists(cls.BOOKING_FILE):
             # Fallback to Existing Booking if the flight one is missing
             if os.path.exists("Existing Booking.xlsx"):
                 cls.BOOKING_FILE = "Existing Booking.xlsx"
             else:
                raise FileNotFoundError(
                    f"Data file '{cls.BOOKING_FILE}' not found in the current directory"
                )
    
    @classmethod
    def is_configured(cls):
        """Check if basic configuration exists"""
        return True
