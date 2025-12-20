"""
Multi-tenant support for production RAG system.

Provides:
- Tenant management and authentication
- Resource isolation (separate collections or filtered queries)
- API key management
- Usage tracking per tenant
"""

from .manager import TenantManager
from .auth import authenticate_tenant, get_current_tenant
from .models import Tenant, APIKey

__all__ = [
    "TenantManager",
    "authenticate_tenant",
    "get_current_tenant",
    "Tenant",
    "APIKey",
]
