"""
Test script for AWS Bedrock Claude Agent
Tests tool calling, language normalization, and RAG functionality
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_service import get_llm_service
from data_service import get_data_service
from vector_service import get_vector_service
from config import Config

def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def test_query(llm_service, query, description):
    print(f"\n📝 Test: {description}")
    print(f"Query: \"{query}\"")
    print("-" * 70)
    
    result = llm_service.process_query(query)
    
    if result.get("success"):
        print(f"✅ Response:\n{result['response']}")
    else:
        print(f"❌ Error: {result.get('error')}")
    
    print("-" * 70)
    return result

def main():
    print_section("AWS BEDROCK CLAUDE AGENT TEST")
    
    # 1. Initialize services
    print("\n🔧 Initializing services...")
    try:
        Config.validate()
        print("✓ Config validated")
        
        vector_service = get_vector_service()
        vector_service.initialize_knowledge_base()
        print("✓ Vector store initialized")
        
        llm_service = get_llm_service()
        print("✓ LLM service initialized (AWS Bedrock Claude)")
        
        data_service = get_data_service()
        print(f"✓ Data service loaded")
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 2. Test RAG Context Retrieval
    print_section("RAG CONTEXT TEST")
    print("Testing vector store knowledge retrieval...")
    docs = vector_service.search("What does HK mean?", k=2)
    for i, doc in enumerate(docs):
        print(f"\nDocument {i+1}:")
        print(doc.page_content[:200] + "...")
    
    # 3. Test Tool Calling
    print_section("TOOL CALLING TESTS")
    
    # Test 1: Status filter - confirmed bookings
    test_query(
        llm_service,
        "Show me all confirmed bookings",
        "Status filter (confirmed → HK)"
    )
    
    # Test 2: Status filter - ticketed
    test_query(
        llm_service,
        "Find ticketed reservations",
        "Status filter (ticketed → TKT)"
    )
    
    # Test 3: Passenger name filter
    test_query(
        llm_service,
        "Find bookings for pax srk",
        "Passenger name extraction"
    )
    
    # Test 4: Reference number
    test_query(
        llm_service,
        "Show ref 12345",
        "Reference number extraction"
    )
    
    # Test 5: Combined filters
    test_query(
        llm_service,
        "Confirmed bookings for client ABC",
        "Multiple filters (status + agent_name)"
    )
    
    # Test 6: Vague query
    test_query(
        llm_service,
        "Show me some bookings",
        "Vague query handling"
    )
    
    # 4. Test Knowledge Questions
    print_section("KNOWLEDGE BASE TESTS")
    
    test_query(
        llm_service,
        "What does HK mean?",
        "Status code definition (should use RAG context)"
    )
    
    test_query(
        llm_service,
        "What status codes are available?",
        "List status codes (should use RAG context)"
    )
    
    # 5. Test Error Handling
    print_section("ERROR HANDLING TESTS")
    
    test_query(
        llm_service,
        "Find bookings for ref NONEXISTENT999",
        "No results scenario"
    )
    
    print_section("TEST COMPLETE")
    print("✅ All tests executed. Review results above.")

if __name__ == "__main__":
    main()
