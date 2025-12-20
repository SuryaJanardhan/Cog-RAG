"""
Authentication and authorization for multi-tenant system.
"""

from typing import Optional
from fastapi import Header, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .models import Tenant, APIKey
from .manager import get_tenant_manager, TenantManager


security = HTTPBearer(auto_error=False)


async def authenticate_tenant(
    authorization: Optional[HTTPAuthorizationCredentials] = Depends(security),
    x_api_key: Optional[str] = Header(None),
) -> Tenant:
    """
    Authenticate request and return tenant.
    
    Supports two authentication methods:
    1. Bearer token: Authorization: Bearer <api_key>
    2. API key header: X-API-Key: <api_key>
    
    Args:
        authorization: Authorization header
        x_api_key: X-API-Key header
        
    Returns:
        Authenticated tenant
        
    Raises:
        HTTPException: If authentication fails
    """
    manager = get_tenant_manager()
    
    # Extract API key from header
    api_key = None
    if authorization and authorization.credentials:
        api_key = authorization.credentials
    elif x_api_key:
        api_key = x_api_key
    
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Provide via 'Authorization: Bearer <key>' or 'X-API-Key: <key>' header",
        )
    
    # Validate API key
    tenant = manager.get_tenant_by_api_key(api_key)
    
    if not tenant:
        raise HTTPException(
            status_code=401,
            detail="Invalid or inactive API key",
        )
    
    if not tenant.is_active:
        raise HTTPException(
            status_code=403,
            detail="Tenant account is inactive",
        )
    
    return tenant


async def get_current_tenant(
    tenant: Tenant = Depends(authenticate_tenant),
) -> Tenant:
    """
    Get current authenticated tenant.
    
    Use this as a dependency in API endpoints.
    """
    return tenant


def get_tenant_context(tenant: Tenant) -> dict:
    """
    Get tenant context for operations.
    
    Returns metadata filter or collection name based on isolation mode.
    
    Args:
        tenant: Tenant object
        
    Returns:
        Dictionary with tenant context
    """
    manager = get_tenant_manager()
    
    return {
        "tenant_id": tenant.id,
        "collection_name": manager.get_collection_name(tenant.id),
        "isolation_mode": tenant.isolation_mode,
        "metadata_filter": {"tenant_id": tenant.id} if tenant.isolation_mode == "filter" else None,
    }
