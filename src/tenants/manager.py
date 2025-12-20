"""
Tenant management system.
"""

import hashlib
import secrets
from typing import Optional, List, Dict
from datetime import datetime
import json
from pathlib import Path

from .models import Tenant, APIKey, UsageStats


class TenantManager:
    """
    Manages tenants, API keys, and usage tracking.
    
    In production, this would use a database (PostgreSQL, MongoDB).
    For demo, uses JSON file storage.
    """
    
    def __init__(self, storage_path: str = "./data/tenants.json"):
        """
        Initialize tenant manager.
        
        Args:
            storage_path: Path to tenant data file
        """
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.tenants: Dict[str, Tenant] = {}
        self.api_keys: Dict[str, APIKey] = {}
        self._load_data()
    
    def _load_data(self):
        """Load tenant data from storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    
                # Load tenants
                for tenant_data in data.get('tenants', []):
                    tenant = Tenant(**tenant_data)
                    self.tenants[tenant.id] = tenant
                
                # Load API keys
                for key_data in data.get('api_keys', []):
                    api_key = APIKey(**key_data)
                    self.api_keys[api_key.key] = api_key
                    
            except Exception as e:
                print(f"Error loading tenant data: {e}")
    
    def _save_data(self):
        """Save tenant data to storage."""
        data = {
            'tenants': [tenant.model_dump(mode='json') for tenant in self.tenants.values()],
            'api_keys': [key.model_dump(mode='json') for key in self.api_keys.values()],
        }
        
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def create_tenant(
        self,
        name: str,
        email: str,
        plan: str = "free",
        isolation_mode: str = "collection",
    ) -> Tenant:
        """
        Create a new tenant.
        
        Args:
            name: Tenant name
            email: Tenant email
            plan: Subscription plan
            isolation_mode: Resource isolation strategy
            
        Returns:
            Created tenant
        """
        tenant = Tenant(
            name=name,
            email=email,
            plan=plan,
            isolation_mode=isolation_mode,
        )
        
        # Set collection name based on isolation mode
        if isolation_mode == "collection":
            tenant.collection_name = f"tenant_{tenant.id.replace('-', '_')}"
        
        # Set limits based on plan
        if plan == "pro":
            tenant.max_documents = 10000
            tenant.max_queries_per_day = 10000
        elif plan == "enterprise":
            tenant.max_documents = 100000
            tenant.max_queries_per_day = 100000
        
        self.tenants[tenant.id] = tenant
        self._save_data()
        
        print(f"Created tenant: {tenant.name} (ID: {tenant.id})")
        return tenant
    
    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant by ID."""
        return self.tenants.get(tenant_id)
    
    def list_tenants(self) -> List[Tenant]:
        """List all tenants."""
        return list(self.tenants.values())
    
    def update_tenant(self, tenant_id: str, **kwargs) -> Optional[Tenant]:
        """Update tenant properties."""
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return None
        
        for key, value in kwargs.items():
            if hasattr(tenant, key):
                setattr(tenant, key, value)
        
        self._save_data()
        return tenant
    
    def delete_tenant(self, tenant_id: str) -> bool:
        """Delete a tenant and all associated API keys."""
        if tenant_id not in self.tenants:
            return False
        
        # Delete tenant
        del self.tenants[tenant_id]
        
        # Delete associated API keys
        keys_to_delete = [
            key for key, api_key in self.api_keys.items()
            if api_key.tenant_id == tenant_id
        ]
        for key in keys_to_delete:
            del self.api_keys[key]
        
        self._save_data()
        return True
    
    def create_api_key(
        self,
        tenant_id: str,
        name: str,
        rate_limit_per_minute: int = 60,
    ) -> tuple[APIKey, str]:
        """
        Create API key for tenant.
        
        Args:
            tenant_id: Tenant ID
            name: Friendly name for key
            rate_limit_per_minute: Rate limit
            
        Returns:
            Tuple of (APIKey object, raw key string)
        """
        if tenant_id not in self.tenants:
            raise ValueError(f"Tenant {tenant_id} not found")
        
        # Generate random API key
        raw_key = f"rag_{secrets.token_urlsafe(32)}"
        
        # Hash the key for storage
        hashed_key = self._hash_key(raw_key)
        
        api_key = APIKey(
            tenant_id=tenant_id,
            key=hashed_key,
            name=name,
            rate_limit_per_minute=rate_limit_per_minute,
        )
        
        self.api_keys[hashed_key] = api_key
        self._save_data()
        
        print(f"Created API key: {name} for tenant {tenant_id}")
        print(f"Key: {raw_key}")
        print("⚠️  Save this key - it won't be shown again!")
        
        return api_key, raw_key
    
    def validate_api_key(self, raw_key: str) -> Optional[APIKey]:
        """
        Validate API key and return associated APIKey object.
        
        Args:
            raw_key: Raw API key string
            
        Returns:
            APIKey object if valid, None otherwise
        """
        hashed_key = self._hash_key(raw_key)
        api_key = self.api_keys.get(hashed_key)
        
        if api_key and api_key.is_active:
            # Update last_used timestamp
            api_key.last_used = datetime.utcnow()
            self._save_data()
            return api_key
        
        return None
    
    def revoke_api_key(self, raw_key: str) -> bool:
        """Revoke an API key."""
        hashed_key = self._hash_key(raw_key)
        api_key = self.api_keys.get(hashed_key)
        
        if api_key:
            api_key.is_active = False
            self._save_data()
            return True
        
        return False
    
    def list_tenant_keys(self, tenant_id: str) -> List[APIKey]:
        """List all API keys for a tenant."""
        return [
            key for key in self.api_keys.values()
            if key.tenant_id == tenant_id
        ]
    
    def get_collection_name(self, tenant_id: str) -> str:
        """
        Get collection name for tenant.
        
        Args:
            tenant_id: Tenant ID
            
        Returns:
            Collection name
        """
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            raise ValueError(f"Tenant {tenant_id} not found")
        
        if tenant.isolation_mode == "collection":
            return tenant.collection_name or f"tenant_{tenant_id.replace('-', '_')}"
        else:
            # Shared collection with filtering
            return "shared_documents"
    
    @staticmethod
    def _hash_key(key: str) -> str:
        """Hash API key for secure storage."""
        return hashlib.sha256(key.encode()).hexdigest()
    
    def get_tenant_by_api_key(self, raw_key: str) -> Optional[Tenant]:
        """Get tenant associated with API key."""
        api_key = self.validate_api_key(raw_key)
        if api_key:
            return self.get_tenant(api_key.tenant_id)
        return None


# Global tenant manager instance
_tenant_manager: Optional[TenantManager] = None


def get_tenant_manager() -> TenantManager:
    """Get global tenant manager instance."""
    global _tenant_manager
    if _tenant_manager is None:
        _tenant_manager = TenantManager()
    return _tenant_manager
