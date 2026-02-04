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
    
    # OpenAI API Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = "gpt-4-0125-preview"  # Latest GPT-4 Turbo with function calling
    OPENAI_TEMPERATURE = 0.1  # Low temperature for consistent, factual responses
    
    # Data File Paths
    # We will use 'Existing Booking.xlsx' as the primary source for this back-office tool
    BOOKING_FILE = "Existing Booking.xlsx"
    
    # System Settings
    MAX_TOKENS = 1000
    TIMEOUT_SECONDS = 30
    
    # Logging
    LOG_LEVEL = "INFO"
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if not cls.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY not found. Please set it in your environment variables or .env file"
            )
        
        if not os.path.exists(cls.BOOKING_FILE):
             # Fallback to Existing Booking if the flight one is missing (though we checked it exists)
             if os.path.exists("Existing Booking.xlsx"):
                 cls.BOOKING_FILE = "Existing Booking.xlsx"
             else:
                raise FileNotFoundError(
                    f"Data file '{cls.BOOKING_FILE}' not found in the current directory"
                )
    
    @classmethod
    def is_configured(cls):
        """Check if API key is configured"""
        return bool(cls.OPENAI_API_KEY)
