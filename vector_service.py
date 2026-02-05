from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from config import Config
import os
import shutil

class VectorService:
    """
    Service to handle interactions with ChromaDB (Vector Database)
    """
    
    def __init__(self):
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL
        )
        self.persist_directory = Config.CHROMA_DB_DIR
        
        # Initialize the DB client
        self.vector_store = Chroma(
            collection_name="travel_knowledge",
            embedding_function=self.embedding_function,
            persist_directory=self.persist_directory
        )
        
    def initialize_knowledge_base(self):
        """
        Populate the vector DB with initial static knowledge (Status codes, etc.)
        This should be run on startup if the DB is empty.
        """
        # Check if we should re-index or if it's already populated
        # For simplicity in this demo, we'll clear and re-add if needed, 
        # but practically we might check document counts.
        
        try:
            # Simple check: try to fetch 1 item
            existing = self.vector_store.get(limit=1)
            if existing and existing['ids']:
                print("Vector DB already populated.")
                return
        except:
            pass
            
        print("Initializing Knowledge Base...")
        
        # Define static knowledge documents
        documents = [
            Document(
                page_content="The status code HK stands for Confirmed. It means the booking is fully confirmed.",
                metadata={"type": "status_code", "code": "HK", "label": "Confirmed"}
            ),
            Document(
                page_content="The status code UC stands for On Request for Flights. It means the flight is not yet confirmed and is pending airline approval.",
                metadata={"type": "status_code", "code": "UC", "label": "On Request (Flight)"}
            ),
            Document(
                page_content="The status code CL stands for Cancelled. It means the booking has been cancelled.",
                metadata={"type": "status_code", "code": "CL", "label": "Cancelled"}
            ),
            Document(
                page_content="The status code TKT stands for Ticketed. It means the ticket has been issued for the flight.",
                metadata={"type": "status_code", "code": "TKT", "label": "Ticketed"}
            ),
            Document(
                page_content="The status code RQ stands for On Request for Hotels. It means the hotel reservation is pending confirmation.",
                metadata={"type": "status_code", "code": "RQ", "label": "On Request (Hotel)"}
            ),
            Document(
                page_content="When a user asks for 'confirmed' bookings, they are looking for status HK.",
                metadata={"type": "rule", "keywords": "confirmed"}
            ),
            Document(
                page_content="When a user asks for 'ticketed' bookings, they are looking for status TKT.",
                metadata={"type": "rule", "keywords": "ticketed"}
            ),
            Document(
                page_content="When a user asks for 'cancelled' bookings, they are looking for status CL.",
                metadata={"type": "rule", "keywords": "cancelled"}
            )
        ]
        
        self.vector_store.add_documents(documents)
        print("Knowledge Base Initialized.")

    def search(self, query: str, k: int = 3):
        """
        Search the vector DB for relevant documents.
        """
        results = self.vector_store.similarity_search(query, k=k)
        return results

# Singleton instance
_vector_service_instance = None

def get_vector_service() -> VectorService:
    global _vector_service_instance
    if _vector_service_instance is None:
        _vector_service_instance = VectorService()
    return _vector_service_instance
