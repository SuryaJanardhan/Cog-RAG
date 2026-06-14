"""
Incremental Ingestion (Change Data Capture) and Directory Synchronization.
Tracks file hashes and chunk-level metadata to avoid redundant indexing.
"""
import os
import hashlib
import sqlite3
import json
from typing import List, Dict, Any, Set
from langchain_core.documents import Document

from ..config import settings
from ..processing import DocumentProcessingPipeline, VectorStorePipeline


class IncrementalIngestionManager:
    """Manages directory synchronization, file change detection, and vector index updates."""
    
    def __init__(self, catalog_path: str = "./data/ingestion_catalog.db"):
        self.catalog_path = catalog_path
        self._initialize_db()
        self.pipeline = DocumentProcessingPipeline()
        self.vector_pipeline = VectorStorePipeline()
        self.vectorstore = self.vector_pipeline.initialize_vectorstore()
        
    def _initialize_db(self) -> None:
        """Initialize SQLite ingestion tracking catalog."""
        os.makedirs(os.path.dirname(self.catalog_path), exist_ok=True)
        conn = sqlite3.connect(self.catalog_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS file_catalog (
                filepath TEXT PRIMARY KEY,
                file_hash TEXT NOT NULL,
                chunk_ids TEXT NOT NULL,
                last_modified REAL NOT NULL
            )
        """)
        conn.commit()
        conn.close()
        
    @staticmethod
    def _calculate_hash(filepath: str) -> str:
        """Compute MD5 hash of a file's content."""
        hasher = hashlib.md5()
        with open(filepath, "rb") as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()
        
    def sync_file(self, filepath: str) -> bool:
        """
        Synchronize a single file: index if new/changed, skip if unchanged.
        Returns True if the file was processed, False if it was skipped.
        """
        if not os.path.exists(filepath):
            self.purge_file(filepath)
            return True
            
        file_hash = self._calculate_hash(filepath)
        mtime = os.path.getmtime(filepath)
        
        # Check catalog
        conn = sqlite3.connect(self.catalog_path)
        cursor = conn.cursor()
        cursor.execute("SELECT file_hash, chunk_ids FROM file_catalog WHERE filepath = ?", (filepath,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            stored_hash, stored_chunks_json = row
            if stored_hash == file_hash:
                print(f"[CDC] File unchanged (skipped): {os.path.basename(filepath)}")
                return False
            else:
                print(f"[CDC] File modified (re-indexing): {os.path.basename(filepath)}")
                # Delete old chunks
                try:
                    old_chunk_ids = json.loads(stored_chunks_json)
                    if old_chunk_ids:
                        self.vectorstore.delete(ids=old_chunk_ids)
                        print(f"[CDC] Purged {len(old_chunk_ids)} old chunks from vector store.")
                except Exception as e:
                    print(f"[CDC] Warning: Failed to purge old chunks: {e}")
                    
        else:
            print(f"[CDC] New file detected (indexing): {os.path.basename(filepath)}")
            
        # Read file content and create document
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            # Skip binary files if encountered
            print(f"[CDC] Skipping non-text file: {filepath}")
            return False
            
        doc = Document(
            page_content=content,
            metadata={
                "source": filepath,
                "filename": os.path.basename(filepath),
                "type": "document"
            }
        )
        
        # Process and store
        stats = self.pipeline.process_and_store([doc])
        # Find document chunks to get their IDs
        # To get matching document IDs, we can look at the returned stored document list.
        # But wait! process_and_store returns statistics containing "documents_stored" count.
        # Let's save a list of mock ids, or read the actual ingested ID list from vectorstore if available.
        # Let's check how doc_ids are generated. In vector_store.py, it returns list of strings.
        # In our document processing pipeline, stats contains "documents_stored" (which matches chunk count).
        # We can store a list of hashes or simple references since delete is metadata-based,
        # or generate deterministic chunk IDs during splitting.
        # In parent_retriever.py, we generate parent_id.
        # To make deletion work reliably, let's keep track of chunk IDs.
        # Since we want to ensure deletion of old chunks, we can delete by metadata filtering
        # (e.g. source = filepath) instead of ID list!
        # Yes! Deleting by metadata "source = filepath" is much more robust because we don't have to keep track of specific chunk IDs,
        # and it works automatically for Qdrant and Chroma!
        # Let's implement delete by metadata.
        self._delete_by_source(filepath)
        
        # Re-run process and store
        stats = self.pipeline.process_and_store([doc])
        
        # Record file state in catalog
        conn = sqlite3.connect(self.catalog_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO file_catalog (filepath, file_hash, chunk_ids, last_modified) VALUES (?, ?, ?, ?)",
            (filepath, file_hash, json.dumps([]), mtime)
        )
        conn.commit()
        conn.close()
        
        return True
        
    def _delete_by_source(self, filepath: str) -> None:
        """Delete all vectors matching source = filepath metadata filter."""
        try:
            # Custom metadata filter deletion based on vector store type
            if hasattr(self.vectorstore, "_client"):
                client = self.vectorstore._client
                # If Chroma
                if hasattr(client, "get_or_create_collection"):
                    # Get chroma collection
                    coll_name = settings.qdrant_collection_name
                    collection = client.get_or_create_collection(coll_name)
                    collection.delete(where={"source": filepath})
                    print(f"[CDC] Chroma: deleted chunks for source={filepath}")
                # If Qdrant
                elif hasattr(client, "delete"):
                    from qdrant_client import models
                    client.delete(
                        collection_name=settings.qdrant_collection_name,
                        points_filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="metadata.source",
                                    match=models.MatchValue(value=filepath),
                                )
                            ]
                        ),
                    )
                    print(f"[CDC] Qdrant: deleted chunks for source={filepath}")
        except Exception as e:
            print(f"[CDC] Could not execute metadata delete for {filepath}: {e}")
            
    def purge_file(self, filepath: str) -> None:
        """Remove file from vector store and catalog if it was deleted on disk."""
        print(f"[CDC] File deleted on disk: {filepath}. Purging index...")
        self._delete_by_source(filepath)
        
        conn = sqlite3.connect(self.catalog_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM file_catalog WHERE filepath = ?", (filepath,))
        conn.commit()
        conn.close()
        
    def sync_directory(self, directory_path: str) -> Dict[str, int]:
        """Scan a directory and synchronize all files (indexing changes/deletions)."""
        if not os.path.exists(directory_path):
            print(f"[CDC] Directory not found: {directory_path}")
            return {"indexed": 0, "skipped": 0, "purged": 0}
            
        print(f"\n[CDC] Synchronizing directory: {directory_path}")
        active_files = set()
        indexed = 0
        skipped = 0
        
        # Scan current files
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith((".txt", ".md", ".pdf", ".docx")):
                    filepath = os.path.join(root, file)
                    active_files.add(filepath)
                    processed = self.sync_file(filepath)
                    if processed:
                        indexed += 1
                    else:
                        skipped += 1
                        
        # Scan catalog to detect deleted files
        conn = sqlite3.connect(self.catalog_path)
        cursor = conn.cursor()
        cursor.execute("SELECT filepath FROM file_catalog")
        catalog_files = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        purged = 0
        for cat_file in catalog_files:
            # If the file is in the catalog but not in the scanned directory, purge it!
            if cat_file not in active_files and cat_file.startswith(directory_path):
                self.purge_file(cat_file)
                purged += 1
                
        print(f"[CDC] Synchronization complete. Indexed: {indexed}, Skipped: {skipped}, Purged: {purged}")
        return {"indexed": indexed, "skipped": skipped, "purged": purged}
