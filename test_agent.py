import unittest
from unittest.mock import MagicMock, patch
import json
import os

# Set up mock env
os.environ["CHROMA_DB_DIR"] = "./test_chroma"

from llm_service import LLMService
from vector_service import VectorService
from data_service import DataService

class TestAgentLogic(unittest.TestCase):
    
    @patch('llm_service.ChatOllama')
    @patch('llm_service.get_vector_service')
    @patch('llm_service.create_tool_calling_agent')
    @patch('llm_service.AgentExecutor')
    def test_agent_initialization(self, mock_executor, mock_create_agent, mock_get_vector, mock_chat_ollama):
        """Verify LLM Service initializes with correct components"""
        service = LLMService()
        
        # Check if Ollama client was initialized
        mock_chat_ollama.assert_called()
        
        # Check if tools are defined
        self.assertTrue(len(service.tools) > 0)
        self.assertEqual(service.tools[0].name, "get_bookings")
        
    @patch('llm_service.get_vector_service')
    def test_process_query_flow(self, mock_get_vector):
        """Verify the process_query method calls vector search and agent invoke"""
        
        # Mock Vector Service
        mock_vector_service = MagicMock()
        mock_doc = MagicMock()
        mock_doc.page_content = "Mock Context"
        mock_vector_service.search.return_value = [mock_doc]
        mock_get_vector.return_value = mock_vector_service
        
        # Mock Agent Executor within LLMService
        # We need to instantiate LLMService, but mock the internal objects
        with patch('llm_service.ChatOllama'), \
             patch('llm_service.create_tool_calling_agent'), \
             patch('llm_service.AgentExecutor') as mock_executor_cls:
            
            mock_executor_instance = MagicMock()
            mock_executor_instance.invoke.return_value = {"output": "Mock Response"}
            mock_executor_cls.return_value = mock_executor_instance
            
            service = LLMService()
            
            # Run
            result = service.process_query("status of srk")
            
            # Verify Vector Search
            mock_vector_service.search.assert_called_with("status of srk", k=2)
            
            # Verify Agent Invocation
            mock_executor_instance.invoke.assert_called()
            call_args = mock_executor_instance.invoke.call_args[0][0]
            self.assertEqual(call_args["input"], "status of srk")
            self.assertIn("Mock Context", call_args["context"])
            
            # Verify Result
            self.assertEqual(result["response"], "Mock Response")

if __name__ == '__main__':
    unittest.main()
