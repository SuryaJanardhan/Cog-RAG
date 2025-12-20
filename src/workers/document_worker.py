"""
Document ingestion worker.
"""

import asyncio
from typing import Dict, Any
from pathlib import Path

from .task_queue import Task
from src.ingestion import DocumentIngestion
from src.tenants import get_tenant_manager


class DocumentWorker:
    """Handles background document ingestion tasks."""
    
    def __init__(self):
        """Initialize document worker."""
        self.ingestion = DocumentIngestion()
    
    async def ingest_document(self, task: Task) -> Dict[str, Any]:
        """
        Ingest a document in the background.
        
        Task data should contain:
        - file_path: Path to document
        - doc_type: Document type (pdf, web, text, etc.)
        - metadata: Optional metadata
        
        Returns:
            Result dictionary with document info
        """
        file_path = task.data.get("file_path")
        doc_type = task.data.get("doc_type", "text")
        metadata = task.data.get("metadata", {})
        
        if not file_path:
            raise ValueError("Missing file_path in task data")
        
        # Add tenant ID to metadata
        metadata["tenant_id"] = task.tenant_id
        
        # Update progress
        task.progress = 10
        task.progress_message = "Loading document..."
        
        # Load document based on type
        if doc_type == "pdf":
            documents = self.ingestion.load_pdf(file_path, metadata)
        elif doc_type == "web":
            documents = self.ingestion.load_web_pages([file_path], metadata)
        elif doc_type == "word":
            documents = self.ingestion.load_word_doc(file_path, metadata)
        else:
            documents = self.ingestion.load_text_file(file_path, metadata)
        
        task.progress = 50
        task.progress_message = f"Loaded {len(documents)} documents"
        
        # Simulate async processing
        await asyncio.sleep(0.1)
        
        task.progress = 100
        task.progress_message = "Ingestion complete"
        
        return {
            "success": True,
            "document_count": len(documents),
            "file_path": str(file_path),
            "doc_type": doc_type,
        }
    
    async def ingest_batch(self, task: Task) -> Dict[str, Any]:
        """
        Ingest multiple documents.
        
        Task data should contain:
        - files: List of file paths
        - doc_types: List of document types (or single type for all)
        """
        files = task.data.get("files", [])
        doc_types = task.data.get("doc_types", "text")
        
        if isinstance(doc_types, str):
            doc_types = [doc_types] * len(files)
        
        results = []
        total = len(files)
        
        for i, (file_path, doc_type) in enumerate(zip(files, doc_types)):
            try:
                # Create sub-task data
                sub_task = Task(
                    type="ingest_document",
                    tenant_id=task.tenant_id,
                    data={
                        "file_path": file_path,
                        "doc_type": doc_type,
                    }
                )
                
                result = await self.ingest_document(sub_task)
                results.append(result)
                
                # Update progress
                task.progress = int((i + 1) / total * 100)
                task.progress_message = f"Processed {i+1}/{total} files"
                
            except Exception as e:
                results.append({
                    "success": False,
                    "file_path": file_path,
                    "error": str(e),
                })
        
        successful = sum(1 for r in results if r.get("success"))
        
        return {
            "success": True,
            "total_files": total,
            "successful": successful,
            "failed": total - successful,
            "results": results,
        }


def create_document_worker() -> DocumentWorker:
    """Factory function to create document worker."""
    return DocumentWorker()
