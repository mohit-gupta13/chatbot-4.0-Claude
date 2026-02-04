"""
LLM Service Module
Handles OpenAI function calling for intent understanding
NO data access - only decides what function to call and with what parameters
"""

from openai import OpenAI
from typing import Dict, Any, Optional, List
import json
from config import Config

class LLMService:
    """
    LLM integration for intent understanding and function calling.
    This module ONLY communicates with the LLM - no data access.
    """
    
    # System prompt that enforces function calling
    SYSTEM_PROMPT = """You are an advanced back-office assistant for a travel agency portal.
You help staff query flight and hotel bookings from the central database.

AVAILABLE STATUS CODES:
- HK: Confirmed
- UC: On Request (Flight)
- CL: Cancelled
- TKT: Ticketed
- RQ: On Request (Hotel)

STRICT RULES:
1. You MUST use the 'get_bookings' function to answer searches. NEVER generate or make up booking data.
2. If a user asks for bookings, reservations, or details about a client/passenger, call 'get_bookings'.
3. Extract as many parameters as possible from the query (e.g., Pax Name, Ref No, Status, City, Client Name, Cancellation Deadline).
4. If the user query is very vague, call 'get_bookings' with no arguments to show recent data, or ask for clarification if needed.
5. Do NOT fabricate data. If the data service returns no results, state that clearly.
6. Understand informal language:
   - "confirmed" or "HK" -> status: "HK"
   - "ticketed" or "TKT" -> status: "TKT"
   - "pax srk" -> pax_name: "srk"
   - "ref 123" -> ref_no: "123"

If the user greets you, respond briefly and offer help with the back-office queries.
For data requests, you MUST call the function.
"""
    
    # Function schemas for OpenAI function calling
    FUNCTION_SCHEMAS = [
        {
            "name": "get_bookings",
            "description": "Search and retrieve travel bookings based on filters like status, passenger name, reference number, or client name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "The status code or label (HK, UC, CL, TKT, RQ, Confirmed, Cancelled, etc.)"
                    },
                    "agent_name": {
                        "type": "string",
                        "description": "The client name or agency name (Client Name field)"
                    },
                    "pax_name": {
                        "type": "string",
                        "description": "The name of the lead passenger (Lead Pax Name field)"
                    },
                    "ref_no": {
                        "type": "string",
                        "description": "The booking reference number (Reference No field)"
                    },
                    "city": {
                        "type": "string",
                        "description": "The city associated with the booking"
                    },
                    "cancellation_deadline": {
                        "type": "string",
                        "description": "The date when the booking will be cancelled (Cancellation Deadline field)"
                    }
                },
                "required": []
            }
        }
    ]
    
    def __init__(self):
        """Initialize OpenAI client"""
        if not Config.is_configured():
            raise ValueError("OPENAI_API_KEY not found.")
        
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.model = Config.OPENAI_MODEL
    
    def process_query(self, user_message: str) -> Dict[str, Any]:
        """
        Process user query and determine intent + function to call
        """
        try:
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                functions=self.FUNCTION_SCHEMAS,
                function_call="auto",
                temperature=Config.OPENAI_TEMPERATURE,
                max_tokens=Config.MAX_TOKENS
            )
            
            message = response.choices[0].message
            
            if message.function_call:
                function_name = message.function_call.name
                function_args = json.loads(message.function_call.arguments)
                
                return {
                    "type": "function_call",
                    "function_name": function_name,
                    "function_args": function_args,
                    "success": True
                }
            else:
                return {
                    "type": "text_response",
                    "response": message.content,
                    "success": True
                }
                
        except Exception as e:
            return {
                "type": "error",
                "error": f"LLM error: {str(e)}",
                "success": False
            }
    
    def format_data_response(
        self, 
        function_name: str, 
        function_args: Dict[str, Any],
        data_result: Dict[str, Any]
    ) -> str:
        """Generate a natural language summary of the results"""
        if not data_result.get("success"):
            return f"Error: {data_result.get('error')}"
        
        count = data_result.get("count", 0)
        display_count = data_result.get("display_count", 0)
        
        if count == 0:
            return "I couldn't find any bookings matching your request."
        
        summary = f"I found {count} matching booking(s)."
        if count > display_count:
            summary += f" Showing the first {display_count}."
            
        filters = [f"{k}: {v}" for k, v in function_args.items() if v]
        if filters:
            summary += " Applied filters: " + ", ".join(filters)
            
        return summary

# Singleton instance
_llm_service_instance = None

def get_llm_service() -> LLMService:
    global _llm_service_instance
    if _llm_service_instance is None:
        _llm_service_instance = LLMService()
    return _llm_service_instance
