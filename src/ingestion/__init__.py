"""Ingestion module for document loading and processing."""
from .document_loader import DocumentIngestion, create_ingestion_pipeline

__all__ = ["DocumentIngestion", "create_ingestion_pipeline"]
