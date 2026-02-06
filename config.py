"""
Configuration module for GenAI Travel Chatbot
Manages AWS Bedrock credentials, Chroma DB, and system settings
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration settings for the chatbot"""
    
    # AWS Bedrock Configuration
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-sonnet-4-20250514")
    
    # Vector DB Configuration
    CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
    
    # Data File Paths
    BOOKING_FILE = os.getenv("BOOKING_FILE", "Existing Booking.xlsx")
    KNOWLEDGE_BASE_FILE = "knowledge_base.txt"
    
    # System Settings
    TIMEOUT_SECONDS = 60
    MAX_TOKENS = 2000
    TEMPERATURE = 0.1  # Low temperature for consistent, factual responses
    
    # Logging
    LOG_LEVEL = "INFO"
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        # Check AWS credentials
        if not cls.AWS_ACCESS_KEY_ID or cls.AWS_ACCESS_KEY_ID == "your-aws-access-key-id-here":
            print("WARNING: AWS_ACCESS_KEY_ID not configured. Using default AWS credentials chain.")
        
        if not cls.AWS_SECRET_ACCESS_KEY or cls.AWS_SECRET_ACCESS_KEY == "your-aws-secret-access-key-here":
            print("WARNING: AWS_SECRET_ACCESS_KEY not configured. Using default AWS credentials chain.")
        
        # Check booking file
        if not os.path.exists(cls.BOOKING_FILE):
            # Try fallback
            if os.path.exists("flight Booking.xlsx"):
                cls.BOOKING_FILE = "flight Booking.xlsx"
            else:
                raise FileNotFoundError(
                    f"Data file '{cls.BOOKING_FILE}' not found in the current directory"
                )
    
    @classmethod
    def is_configured(cls):
        """Check if basic configuration exists"""
        return True  # Will use AWS credentials chain if env vars not set
