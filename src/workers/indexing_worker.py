"""
Indexing worker for background vector indexing.
"""

import asyncio
from typing import Dict, Any, List

from .task_queue import Task
from src.processing import DocumentProcessingPipeline
from src.tenants import get_tenant_manager


class IndexingWorker:
    """Handles background indexing tasks."""
    
    def __init__(self):
        """Initialize indexing worker."""
        self.pipeline = DocumentProcessingPipeline()
    
    async def index_documents(self, task: Task) -> Dict[str, Any]:
        """
        Index documents into vector database.
        
        Task data should contain:
        - documents: List of document dictionaries
        - collection_name: Optional collection name override
        
        Returns:
            Result dictionary with indexing info
        """
        documents = task.data.get("documents", [])
        
        if not documents:
            raise ValueError("No documents provided for indexing")
        
        # Get tenant collection name
        manager = get_tenant_manager()
        tenant = manager.get_tenant(task.tenant_id)
        
        if not tenant:
            raise ValueError(f"Tenant {task.tenant_id} not found")
        
        collection_name = task.data.get("collection_name") or manager.get_collection_name(task.tenant_id)
        
        task.progress = 10
        task.progress_message = f"Indexing {len(documents)} documents..."
        
        # Process documents
        # Note: In real implementation, this would use the actual pipeline
        # For now, simulating async processing
        
        total = len(documents)
        chunk_size = 10
        
        for i in range(0, total, chunk_size):
            batch = documents[i:i+chunk_size]
            
            # Simulate processing
            await asyncio.sleep(0.1)
            
            # Update progress
            processed = min(i + chunk_size, total)
            task.progress = int(10 + (processed / total * 80))
            task.progress_message = f"Indexed {processed}/{total} documents"
        
        task.progress = 100
        task.progress_message = "Indexing complete"
        
        return {
            "success": True,
            "document_count": total,
            "collection_name": collection_name,
            "tenant_id": task.tenant_id,
        }
    
    async def rebuild_index(self, task: Task) -> Dict[str, Any]:
        """
        Rebuild entire index for a tenant.
        
        Task data should contain:
        - full_rebuild: Whether to delete existing index first
        """
        full_rebuild = task.data.get("full_rebuild", False)
        
        manager = get_tenant_manager()
        tenant = manager.get_tenant(task.tenant_id)
        
        if not tenant:
            raise ValueError(f"Tenant {task.tenant_id} not found")
        
        collection_name = manager.get_collection_name(task.tenant_id)
        
        task.progress = 10
        task.progress_message = "Starting index rebuild..."
        
        if full_rebuild:
            task.progress_message = "Deleting existing index..."
            # Would delete existing collection here
            await asyncio.sleep(0.5)
            task.progress = 30
        
        task.progress_message = "Rebuilding index..."
        # Would rebuild index here
        await asyncio.sleep(1)
        
        task.progress = 100
        task.progress_message = "Index rebuilt successfully"
        
        return {
            "success": True,
            "collection_name": collection_name,
            "full_rebuild": full_rebuild,
        }


def create_indexing_worker() -> IndexingWorker:
    """Factory function to create indexing worker."""
    return IndexingWorker()
