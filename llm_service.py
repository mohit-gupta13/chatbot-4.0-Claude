"""
LLM Service Module
Handles AWS Bedrock Claude integration with LangChain agent
Implements tool calling for booking retrieval with RAG context
"""

from datetime import datetime
from typing import Dict, Any, Optional
from langchain_aws import ChatBedrock
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from config import Config
from data_service import get_data_service
from vector_service import get_vector_service
import json
import boto3

# Define the get_bookings tool using LangChain's @tool decorator
@tool
def get_bookings(
    status: Optional[str] = None,
    agent_name: Optional[str] = None,
    pax_name: Optional[str] = None,
    ref_no: Optional[str] = None,
    city: Optional[str] = None,
    cancellation_deadline: Optional[str] = None
) -> str:
    """
    Search and retrieve travel bookings from the database.
    
    Use this tool to find flight and hotel bookings based on filters:
    - status: Booking status code (HK=Confirmed, UC=On Request Flight, CL=Cancelled, TKT=Ticketed, RQ=On Request Hotel)
    - agent_name: Client name or agency name
    - pax_name: Lead passenger name
    - ref_no: Booking reference number
    - city: City associated with the booking
    - cancellation_deadline: Cancellation deadline date
    
    Always use this tool when users ask about bookings. Never fabricate booking data.
    """
    data_service = get_data_service()
    result = data_service.get_bookings(
        status=status,
        agent_name=agent_name,
        pax_name=pax_name,
        ref_no=ref_no,
        city=city,
        cancellation_deadline=cancellation_deadline
    )
    return json.dumps(result, default=str)

class LLMService:
    """
    LLM integration using AWS Bedrock Claude with LangChain.
    Manages the ReAct agent with tool calling and RAG.
    """
    
    SYSTEM_TEMPLATE = """You are an advanced back-office assistant for a B2B travel agency portal.
You help travel agents query flight and hotel bookings from the central database using natural language.

BOOKING STATUS CODES (from Knowledge Base):
{context}

STRICT BEHAVIOR RULES:
1. You MUST use the 'get_bookings' tool to answer any booking-related queries. NEVER generate or fabricate booking data.
2. When a user asks for bookings, reservations, tickets, or details about a client/passenger/city, call 'get_bookings'.
3. Extract as many filter parameters as possible from the user's natural language query before calling the tool.
4. If a query is vague (e.g., "show me bookings"), call 'get_bookings' with no arguments to show recent data, or ask for clarification if appropriate.
5. If the tool returns no results, clearly state "No bookings found matching your criteria."
6. NEVER invent or make up booking information. Only use data returned by the tool.

LANGUAGE NORMALIZATION:
Understand informal language and map it to proper filters:
- "confirmed" or "HK" → status: "HK"
- "ticketed" or "TKT" → status: "TKT"
- "cancelled" or "CL" → status: "CL"
- "on request flight" or "UC" → status: "UC"
- "on request hotel" or "RQ" → status: "RQ"
- "pax srk" → pax_name: "srk"
- "ref 123" → ref_no: "123"
- "client ABC" → agent_name: "ABC"

RESPONSE STYLE:
- Be professional and helpful
- Provide clear, concise summaries of booking data
- If multiple bookings are found, mention the count
- Always acknowledge what filters were applied

Current Time: {time}
"""
    
    def __init__(self):
        """Initialize AWS Bedrock Claude with LangChain agent"""
        
        # Initialize AWS Bedrock client
        bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name=Config.AWS_REGION,
            aws_access_key_id=Config.AWS_ACCESS_KEY_ID if Config.AWS_ACCESS_KEY_ID else None,
            aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY if Config.AWS_SECRET_ACCESS_KEY else None
        )
        
        # Initialize Claude via Bedrock
        self.llm = ChatBedrock(
            client=bedrock_runtime,
            model_id=Config.BEDROCK_MODEL_ID,
            model_kwargs={
                "temperature": Config.TEMPERATURE,
                "max_tokens": Config.MAX_TOKENS,
            }
        )
        
        # Define available tools
        self.tools = [get_bookings]
        
        # Initialize vector service for RAG
        self.vector_service = get_vector_service()
        
        # Define the prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_TEMPLATE),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        # Create the tool-calling agent
        self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
    
    def process_query(self, user_message: str) -> Dict[str, Any]:
        """
        Process user query through the LangChain Agent with RAG.
        
        Args:
            user_message: Natural language query from user
            
        Returns:
            Dictionary with success status and response
        """
        try:
            # 1. Retrieve context from Vector DB (RAG)
            docs = self.vector_service.search(user_message, k=3)
            context_text = "\n".join([d.page_content for d in docs])
            
            if not context_text:
                context_text = "No additional context available from knowledge base."
            
            # 2. Invoke LangChain Agent
            response = self.agent_executor.invoke({
                "input": user_message,
                "context": context_text,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # 3. Return response
            return {
                "success": True,
                "response": response["output"]
            }
                
        except Exception as e:
            print(f"LLM/Agent Error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": f"Failed to process query: {str(e)}"
            }

# Singleton instance
_llm_service_instance = None

def get_llm_service() -> LLMService:
    """Get or create singleton LLM service instance"""
    global _llm_service_instance
    if _llm_service_instance is None:
        _llm_service_instance = LLMService()
    return _llm_service_instance
