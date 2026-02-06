"""
Vector Service Module
Manages Chroma vector database with local Sentence-Transformers embeddings
Provides RAG context retrieval for the LLM
"""

import os
from typing import List, Optional
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
from config import Config

class VectorService:
    """
    Vector database service for RAG.
    Uses Chroma with local Sentence-Transformers embeddings.
    """
    
    def __init__(self):
        """Initialize vector service with Chroma and local embeddings"""
        print(f"Initializing embeddings model: {Config.EMBEDDING_MODEL}")
        
        # Initialize local embeddings (Sentence-Transformers)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},  # Use 'cuda' if GPU available
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize or load Chroma vector store
        self.vector_store = None
        self._initialized = False
    
    def initialize_knowledge_base(self):
        """
        Load knowledge base from text file and populate Chroma DB.
        This should be called once on startup.
        """
        if self._initialized:
            print("Vector store already initialized")
            return
        
        try:
            # Check if knowledge base file exists
            if not os.path.exists(Config.KNOWLEDGE_BASE_FILE):
                print(f"WARNING: Knowledge base file '{Config.KNOWLEDGE_BASE_FILE}' not found")
                # Create empty vector store
                self.vector_store = Chroma(
                    collection_name="booking_knowledge",
                    embedding_function=self.embeddings,
                    persist_directory=Config.CHROMA_DB_DIR
                )
                self._initialized = True
                return
            
            # Load knowledge base content
            with open(Config.KNOWLEDGE_BASE_FILE, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split content into chunks
            text_splitter = CharacterTextSplitter(
                separator="\n\n",
                chunk_size=500,
                chunk_overlap=50,
                length_function=len
            )
            
            texts = text_splitter.split_text(content)
            
            # Create documents
            documents = [Document(page_content=text) for text in texts]
            
            # Check if vector store already exists
            if os.path.exists(Config.CHROMA_DB_DIR):
                print(f"Loading existing Chroma DB from {Config.CHROMA_DB_DIR}")
                self.vector_store = Chroma(
                    collection_name="booking_knowledge",
                    embedding_function=self.embeddings,
                    persist_directory=Config.CHROMA_DB_DIR
                )
            else:
                print(f"Creating new Chroma DB at {Config.CHROMA_DB_DIR}")
                # Create new vector store with documents
                self.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    collection_name="booking_knowledge",
                    persist_directory=Config.CHROMA_DB_DIR
                )
            
            self._initialized = True
            print(f"Vector store initialized with {len(documents)} knowledge chunks")
            
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            # Create empty vector store as fallback
            self.vector_store = Chroma(
                collection_name="booking_knowledge",
                embedding_function=self.embeddings,
                persist_directory=Config.CHROMA_DB_DIR
            )
            self._initialized = True
    
    def search(self, query: str, k: int = 3) -> List[Document]:
        """
        Search for relevant documents in the vector store.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant documents
        """
        if not self._initialized or self.vector_store is None:
            print("WARNING: Vector store not initialized, returning empty results")
            return []
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            return results
        except Exception as e:
            print(f"Error searching vector store: {e}")
            return []
    
    def add_documents(self, texts: List[str]) -> bool:
        """
        Add new documents to the vector store.
        
        Args:
            texts: List of text strings to add
            
        Returns:
            True if successful, False otherwise
        """
        if not self._initialized or self.vector_store is None:
            print("ERROR: Vector store not initialized")
            return False
        
        try:
            documents = [Document(page_content=text) for text in texts]
            self.vector_store.add_documents(documents)
            return True
        except Exception as e:
            print(f"Error adding documents: {e}")
            return False

# Singleton instance
_vector_service_instance = None

def get_vector_service() -> VectorService:
    """Get or create singleton vector service instance"""
    global _vector_service_instance
    if _vector_service_instance is None:
        _vector_service_instance = VectorService()
    return _vector_service_instance
