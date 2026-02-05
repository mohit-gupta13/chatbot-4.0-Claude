from datetime import datetime
from typing import Dict, Any, Optional, List
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from config import Config
from data_service import get_data_service
from vector_service import get_vector_service
import json

# Define the tool using LangChain's @tool decorator
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
    Useful when adding filters like status (HK, UC, CL, TKT, RQ), passenger name, reference number, or client name.
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
    LLM integration using LangChain and Ollama.
    Manages the ReAct / Tool-calling agent.
    """
    
    SYSTEM_TEMPLATE = """You are an advanced back-office assistant for a travel agency portal.
You help staff query flight and hotel bookings from the central database.

AVAILABLE STATUS CODES (from Knowledge Base):
{context}

STRICT RULES:
1. You MUST use the 'get_bookings' tool to answer searches. NEVER generate or make up booking data.
2. If a user asks for bookings, reservations, or details about a client/passenger, call 'get_bookings'.
3. Extract as many parameters as possible from the query (e.g., Pax Name, Ref No, Status, City, Client Name).
4. If the user query is very vague, call 'get_bookings' with no arguments to show recent data.
5. Do NOT fabricate data. If the tool returns no results, state that clearly.
6. Understand informal language:
   - "confirmed" or "HK" -> status: "HK"
   - "ticketed" or "TKT" -> status: "TKT"
   - "pax srk" -> pax_name: "srk"
   - "ref 123" -> ref_no: "123"

Current Time: {time}
"""
    
    def __init__(self):
        self.llm = ChatOllama(
            model=Config.OLLAMA_MODEL,
            base_url=Config.OLLAMA_BASE_URL,
            temperature=0.1,
            keep_alive="5m"
        )
        
        self.tools = [get_bookings]
        self.vector_service = get_vector_service()
        
        # Define the prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_TEMPLATE),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        # Create the agent
        self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent, 
            tools=self.tools, 
            verbose=True,
            handle_parsing_errors=True
        )
    
    def process_query(self, user_message: str) -> Dict[str, Any]:
        """
        Process user query through the LangChain Agent.
        """
        try:
            # 1. Retrieve context from Vector DB
            docs = self.vector_service.search(user_message, k=2)
            context_text = "\n".join([d.page_content for d in docs])
            
            # 2. Invoke Agent
            response = self.agent_executor.invoke({
                "input": user_message,
                "context": context_text,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            return {
                "success": True,
                "type": "text_response", # LangChain handles the final text generation
                "response": response["output"]
            }
                
        except Exception as e:
            print(f"LLM/Agent Error: {e}")
            return {
                "success": False,
                "type": "error",
                "error": str(e)
            }

# Singleton instance
_llm_service_instance = None

def get_llm_service() -> LLMService:
    global _llm_service_instance
    if _llm_service_instance is None:
        _llm_service_instance = LLMService()
    return _llm_service_instance
