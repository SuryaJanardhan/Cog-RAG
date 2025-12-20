"""
Data models for multi-tenant system.
"""

from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field
import uuid


class Tenant(BaseModel):
    """Tenant model."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    email: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True
    plan: str = "free"  # free, pro, enterprise
    max_documents: int = 1000
    max_queries_per_day: int = 1000
    
    # Resource isolation strategy
    isolation_mode: str = "collection"  # collection or filter
    collection_name: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Acme Corp",
                "email": "admin@acme.com",
                "plan": "pro",
                "isolation_mode": "collection",
            }
        }


class APIKey(BaseModel):
    """API key model for tenant authentication."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    key: str  # Hashed API key
    name: str  # Friendly name for the key
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_used: Optional[datetime] = None
    is_active: bool = True
    
    # Rate limiting
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000
    rate_limit_per_day: int = 10000
    
    class Config:
        json_schema_extra = {
            "example": {
                "tenant_id": "tenant-123",
                "name": "Production API Key",
                "rate_limit_per_minute": 100,
            }
        }


class UsageStats(BaseModel):
    """Usage statistics for a tenant."""
    
    tenant_id: str
    date: datetime
    queries_count: int = 0
    documents_indexed: int = 0
    tokens_used: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_response_time_ms: float = 0.0
    errors_count: int = 0
