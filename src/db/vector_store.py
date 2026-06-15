"""
Vector database initialization and management.
Supports Qdrant (prod) and Chroma (dev) based on configuration.
"""
from typing import Optional
from abc import ABC, abstractmethod
try:
    from langchain_qdrant import Qdrant
except ImportError:
    try:
        from langchain_community.vectorstores import Qdrant
    except ImportError:
        Qdrant = None

try:
    from langchain_chroma import Chroma
except ImportError:
    try:
        from langchain_community.vectorstores import Chroma
    except ImportError:
        Chroma = None
from langchain_core.vectorstores import VectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import chromadb
from chromadb.config import Settings as ChromaSettings

from ..config import settings


class VectorDBClient(ABC):
    """Abstract base class for vector database clients."""
    
    @abstractmethod
    def get_vectorstore(self, embeddings) -> VectorStore:
        """Get the vector store instance."""
        pass
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the vector database."""
        pass


class QdrantVectorDB(VectorDBClient):
    """Qdrant vector database client for production."""
    
    def __init__(self):
        self.client: Optional[QdrantClient] = None
        self.collection_name = settings.qdrant_collection_name
        
    def initialize(self) -> None:
        """Initialize Qdrant client and create collection if needed."""
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        
        # Check if collection exists, create if not
        collections = self.client.get_collections().collections
        collection_names = [col.name for col in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=settings.embedding_dimension,
                    distance=Distance.COSINE,
                ),
            )
            print(f"Created Qdrant collection: {self.collection_name}")
        else:
            print(f"Using existing Qdrant collection: {self.collection_name}")
    
    def get_vectorstore(self, embeddings) -> VectorStore:
        """Get Qdrant vector store instance."""
        if not self.client:
            self.initialize()
        
        return Qdrant(
            client=self.client,
            collection_name=self.collection_name,
            embeddings=embeddings,
        )


class ChromaVectorDB(VectorDBClient):
    """Chroma vector database client for development."""
    
    def __init__(self):
        self.persist_directory = settings.chroma_persist_directory
        self.client: Optional[chromadb.Client] = None
        
    def initialize(self) -> None:
        """Initialize Chroma client with persistent storage."""
        self.client = chromadb.Client(
            ChromaSettings(
                persist_directory=self.persist_directory,
                anonymized_telemetry=False,
            )
        )
        print(f"Initialized Chroma DB at: {self.persist_directory}")
    
    def get_vectorstore(self, embeddings) -> VectorStore:
        """Get Chroma vector store instance."""
        if not self.client:
            self.initialize()
        
        return Chroma(
            persist_directory=self.persist_directory,
            embedding_function=embeddings,
            client=self.client,
        )


class PineconeVectorDB(VectorDBClient):
    """Pinecone vector database client for production cloud hosting."""
    
    def __init__(self):
        self.api_key = settings.pinecone_api_key
        self.index_name = settings.pinecone_index_name
        
    def initialize(self) -> None:
        """Initialize Pinecone index client."""
        if not self.api_key:
            print("[Warning] Pinecone API Key is not set.")
            return
            
        try:
            from pinecone import Pinecone, ServerlessSpec
            pc = Pinecone(api_key=self.api_key)
            existing_indexes = [idx.name for idx in pc.list_indexes()]
            if self.index_name not in existing_indexes:
                print(f"Creating Pinecone index: {self.index_name}...")
                pc.create_index(
                    name=self.index_name,
                    dimension=settings.embedding_dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                print(f"Successfully created Pinecone index: {self.index_name}")
            else:
                print(f"Using existing Pinecone index: {self.index_name}")
        except Exception as e:
            print(f"[Pinecone] Warning during initialization check: {e}")
            
    def get_vectorstore(self, embeddings) -> VectorStore:
        """Get Pinecone vector store instance."""
        try:
            from langchain_pinecone import PineconeVectorStore
        except ImportError:
            raise ImportError(
                "Could not import langchain_pinecone. "
                "Please install it with `pip install langchain-pinecone`."
            )
        
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY is required to connect to Pinecone.")
            
        return PineconeVectorStore(
            index_name=self.index_name,
            embedding=embeddings,
            pinecone_api_key=self.api_key
        )


def get_vector_db() -> VectorDBClient:
    """
    Factory function to get the appropriate vector database client
    based on environment configuration.
    
    Returns:
        VectorDBClient: Configured vector database client
    """
    if settings.vector_db == "qdrant":
        print("Using Qdrant vector database (production)")
        return QdrantVectorDB()
    elif settings.vector_db == "pinecone":
        print("Using Pinecone vector database (production cloud)")
        return PineconeVectorDB()
    else:
        print("Using Chroma vector database (development)")
        return ChromaVectorDB()


# Global vector DB instance
vector_db_client: Optional[VectorDBClient] = None


def initialize_vector_db() -> VectorDBClient:
    """Initialize and return the global vector database client."""
    global vector_db_client
    if vector_db_client is None:
        vector_db_client = get_vector_db()
        vector_db_client.initialize()
    return vector_db_client
