"""
Production tenant management demo (Phase 4).

Demonstrates:
- Tenant creation and management
- API key generation
- Multi-tenant isolation
- Background task submission
- Rate limiting behavior
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.tenants import TenantManager, get_tenant_manager
from src.workers import TaskQueue, get_task_queue, create_document_worker, create_indexing_worker


def demo_tenant_management():
    """Demonstrate tenant management."""
    print("\n" + "="*80)
    print("TENANT MANAGEMENT DEMO")
    print("="*80)
    
    manager = get_tenant_manager()
    
    # Create tenants
    print("\n1. Creating tenants...")
    
    tenant1 = manager.create_tenant(
        name="Acme Corporation",
        email="admin@acme.com",
        plan="pro",
        isolation_mode="collection",
    )
    print(f"   ✓ Created: {tenant1.name}")
    print(f"     - ID: {tenant1.id}")
    print(f"     - Plan: {tenant1.plan}")
    print(f"     - Collection: {tenant1.collection_name}")
    print(f"     - Max docs: {tenant1.max_documents}")
    
    tenant2 = manager.create_tenant(
        name="Beta Inc",
        email="admin@beta.com",
        plan="free",
        isolation_mode="collection",
    )
    print(f"   ✓ Created: {tenant2.name}")
    print(f"     - ID: {tenant2.id}")
    print(f"     - Plan: {tenant2.plan}")
    
    # Create API keys
    print("\n2. Generating API keys...")
    
    api_key1, raw_key1 = manager.create_api_key(
        tenant_id=tenant1.id,
        name="Production Key",
        rate_limit_per_minute=100,
    )
    print(f"   ✓ API Key for {tenant1.name}:")
    print(f"     - Name: {api_key1.name}")
    print(f"     - Rate limit: {api_key1.rate_limit_per_minute}/min")
    print(f"     - Key: {raw_key1[:20]}...")
    
    api_key2, raw_key2 = manager.create_api_key(
        tenant_id=tenant2.id,
        name="Development Key",
        rate_limit_per_minute=30,
    )
    print(f"   ✓ API Key for {tenant2.name}:")
    print(f"     - Rate limit: {api_key2.rate_limit_per_minute}/min")
    
    # List tenants
    print("\n3. Listing all tenants...")
    tenants = manager.list_tenants()
    for tenant in tenants:
        print(f"   - {tenant.name} ({tenant.plan})")
    
    # Validate API key
    print("\n4. Validating API keys...")
    validated = manager.validate_api_key(raw_key1)
    if validated:
        print(f"   ✓ Key validated for tenant: {validated.tenant_id}")
    
    # List tenant keys
    print("\n5. Listing tenant keys...")
    keys = manager.list_tenant_keys(tenant1.id)
    print(f"   Tenant {tenant1.name} has {len(keys)} API key(s)")
    
    return tenant1, tenant2, raw_key1, raw_key2


def demo_background_tasks(tenant1, tenant2):
    """Demonstrate background task queue."""
    print("\n" + "="*80)
    print("BACKGROUND TASKS DEMO")
    print("="*80)
    
    queue = get_task_queue()
    
    # Register workers
    print("\n1. Registering task handlers...")
    doc_worker = create_document_worker()
    index_worker = create_indexing_worker()
    
    queue.register_handler("ingest_document", doc_worker.ingest_document)
    queue.register_handler("ingest_batch", doc_worker.ingest_batch)
    queue.register_handler("index_documents", index_worker.index_documents)
    queue.register_handler("rebuild_index", index_worker.rebuild_index)
    
    # Submit tasks
    print("\n2. Submitting tasks...")
    
    task1 = queue.submit_task(
        task_type="ingest_document",
        tenant_id=tenant1.id,
        data={
            "file_path": "./data/raw/sample.txt",
            "doc_type": "text",
            "metadata": {"source": "upload"},
        }
    )
    print(f"   ✓ Submitted task: {task1.id}")
    print(f"     - Type: {task1.type}")
    print(f"     - Status: {task1.status}")
    
    task2 = queue.submit_task(
        task_type="index_documents",
        tenant_id=tenant2.id,
        data={
            "documents": [{"content": "Sample doc"}],
        }
    )
    print(f"   ✓ Submitted task: {task2.id}")
    
    # List tasks
    print("\n3. Listing tasks...")
    tasks = queue.list_tasks()
    print(f"   Total tasks: {len(tasks)}")
    
    tenant1_tasks = queue.list_tasks(tenant_id=tenant1.id)
    print(f"   Tenant {tenant1.id[:8]}... tasks: {len(tenant1_tasks)}")
    
    print("\n4. Task queue summary:")
    print(f"   - Registered handlers: {len(queue.handlers)}")
    print(f"   - Pending tasks: {len(queue.list_tasks(status='pending'))}")
    print(f"   - Total tasks: {len(queue.tasks)}")


def demo_rate_limiting():
    """Demonstrate rate limiting concepts."""
    print("\n" + "="*80)
    print("RATE LIMITING DEMO")
    print("="*80)
    
    from src.api.rate_limit import RateLimiter
    
    limiter = RateLimiter()
    
    print("\n1. Testing rate limits...")
    print("   Simulating 5 requests with 3/minute limit:")
    
    for i in range(5):
        allowed, retry_after = limiter.is_allowed(
            key="test-key",
            limit=3,
            window_seconds=60,
        )
        
        if allowed:
            print(f"   ✓ Request {i+1}: Allowed")
        else:
            print(f"   ✗ Request {i+1}: Blocked (retry after {retry_after}s)")
    
    # Usage stats
    print("\n2. Getting usage stats...")
    usage = limiter.get_usage("test-key", 60)
    print(f"   Requests in window: {usage['requests']}")
    print(f"   Window: {usage['window_seconds']}s")


def demo_observability():
    """Demonstrate observability features."""
    print("\n" + "="*80)
    print("OBSERVABILITY DEMO")
    print("="*80)
    
    from src.api.observability import (
        get_metrics_collector,
        StructuredLogger,
        get_tracer,
    )
    
    # Metrics
    print("\n1. Metrics collection...")
    metrics = get_metrics_collector()
    
    metrics.increment("requests_total")
    metrics.increment("requests_by_tenant", tenant_id="tenant-123")
    metrics.record_response_time(150.5)
    metrics.increment("cache_hits")
    
    current_metrics = metrics.get_metrics()
    print(f"   Total requests: {current_metrics['requests_total']}")
    print(f"   Cache hits: {current_metrics['cache_hits']}")
    print(f"   Avg response time: {current_metrics['avg_response_time_ms']:.2f}ms")
    
    # Logging
    print("\n2. Structured logging...")
    logger = StructuredLogger("demo")
    
    logger.info(
        "Query processed",
        tenant_id="tenant-123",
        query_length=50,
        duration_ms=145,
    )
    print("   ✓ Log entry created")
    
    # Tracing
    print("\n3. Tracing...")
    tracer = get_tracer(enabled=True)
    
    with tracer.trace_chain("test_chain", {"input": "test"}):
        print("   ✓ Chain traced")
    
    with tracer.trace_llm("gemini-1.5-flash", "Test prompt"):
        print("   ✓ LLM call traced")


def demo_resource_isolation():
    """Demonstrate multi-tenant resource isolation."""
    print("\n" + "="*80)
    print("RESOURCE ISOLATION DEMO")
    print("="*80)
    
    manager = get_tenant_manager()
    tenants = manager.list_tenants()
    
    print("\n1. Collection-based isolation:")
    for tenant in tenants[:2]:  # Show first 2
        if tenant.isolation_mode == "collection":
            coll_name = manager.get_collection_name(tenant.id)
            print(f"   {tenant.name}:")
            print(f"     - Collection: {coll_name}")
            print(f"     - Isolated: Yes")
    
    print("\n2. Isolation strategies:")
    print("   ✓ Collection per tenant: Complete isolation")
    print("   ✓ Shared collection + filters: Resource efficient")
    print("   ✓ Configurable per tenant")


def main():
    """Run all demos."""
    print("\n" + "="*80)
    print("PRODUCTION FEATURES DEMO (Phase 4)")
    print("="*80)
    
    try:
        # Tenant management
        tenant1, tenant2, key1, key2 = demo_tenant_management()
        
        # Background tasks
        demo_background_tasks(tenant1, tenant2)
        
        # Rate limiting
        demo_rate_limiting()
        
        # Observability
        demo_observability()
        
        # Resource isolation
        demo_resource_isolation()
        
        print("\n" + "="*80)
        print("DEMO COMPLETE!")
        print("="*80)
        print("\n✅ Phase 4 Features Demonstrated:")
        print("   - Multi-tenant management")
        print("   - API key authentication")
        print("   - Background task queue")
        print("   - Rate limiting")
        print("   - Metrics collection")
        print("   - Structured logging")
        print("   - LangSmith tracing")
        print("   - Resource isolation")
        
    except Exception as e:
        print(f"\nDemo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
