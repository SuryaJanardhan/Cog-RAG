"""
Task queue for background workers.
"""

import asyncio
from typing import Dict, Any, Optional, Callable, Awaitable
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
import uuid
import json
from pathlib import Path


class TaskStatus(str, Enum):
    """Task status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Task(BaseModel):
    """Background task model."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str  # ingest_document, index_documents, etc.
    tenant_id: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Task data
    data: Dict[str, Any] = Field(default_factory=dict)
    
    # Result
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    # Progress tracking
    progress: int = 0  # 0-100
    progress_message: Optional[str] = None


class TaskQueue:
    """
    Simple in-memory task queue.
    
    In production, use Celery, Redis Queue, or cloud task queues.
    """
    
    def __init__(self, storage_path: str = "./data/tasks.json"):
        """Initialize task queue."""
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.tasks: Dict[str, Task] = {}
        self.handlers: Dict[str, Callable] = {}
        self.running = False
        
        self._load_tasks()
    
    def _load_tasks(self):
        """Load tasks from storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    for task_data in data:
                        task = Task(**task_data)
                        self.tasks[task.id] = task
            except Exception as e:
                print(f"Error loading tasks: {e}")
    
    def _save_tasks(self):
        """Save tasks to storage."""
        try:
            data = [task.model_dump(mode='json') for task in self.tasks.values()]
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving tasks: {e}")
    
    def register_handler(
        self,
        task_type: str,
        handler: Callable[[Task], Awaitable[Dict[str, Any]]],
    ):
        """
        Register task handler.
        
        Args:
            task_type: Task type identifier
            handler: Async function that processes the task
        """
        self.handlers[task_type] = handler
        print(f"Registered handler for task type: {task_type}")
    
    def submit_task(
        self,
        task_type: str,
        tenant_id: str,
        data: Dict[str, Any],
    ) -> Task:
        """
        Submit a new task to the queue.
        
        Args:
            task_type: Type of task
            tenant_id: Tenant ID
            data: Task data
            
        Returns:
            Created task
        """
        task = Task(
            type=task_type,
            tenant_id=tenant_id,
            data=data,
        )
        
        self.tasks[task.id] = task
        self._save_tasks()
        
        print(f"Submitted task {task.id} of type {task_type}")
        return task
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        return self.tasks.get(task_id)
    
    def list_tasks(
        self,
        tenant_id: Optional[str] = None,
        status: Optional[TaskStatus] = None,
    ) -> list[Task]:
        """
        List tasks with optional filtering.
        
        Args:
            tenant_id: Filter by tenant
            status: Filter by status
            
        Returns:
            List of tasks
        """
        tasks = list(self.tasks.values())
        
        if tenant_id:
            tasks = [t for t in tasks if t.tenant_id == tenant_id]
        
        if status:
            tasks = [t for t in tasks if t.status == status]
        
        return sorted(tasks, key=lambda t: t.created_at, reverse=True)
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task."""
        task = self.tasks.get(task_id)
        if task and task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            self._save_tasks()
            return True
        return False
    
    async def start_worker(self, num_workers: int = 2):
        """
        Start worker processes.
        
        Args:
            num_workers: Number of concurrent workers
        """
        self.running = True
        print(f"Starting {num_workers} workers...")
        
        workers = [
            self._worker_loop(worker_id=i)
            for i in range(num_workers)
        ]
        
        await asyncio.gather(*workers)
    
    async def _worker_loop(self, worker_id: int):
        """Worker loop that processes tasks."""
        print(f"Worker {worker_id} started")
        
        while self.running:
            # Get next pending task
            task = self._get_next_task()
            
            if task:
                try:
                    await self._process_task(task, worker_id)
                except Exception as e:
                    print(f"Worker {worker_id} error processing task {task.id}: {e}")
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    task.completed_at = datetime.utcnow()
                    self._save_tasks()
            else:
                # No tasks, wait a bit
                await asyncio.sleep(1)
    
    def _get_next_task(self) -> Optional[Task]:
        """Get next pending task."""
        for task in self.tasks.values():
            if task.status == TaskStatus.PENDING:
                return task
        return None
    
    async def _process_task(self, task: Task, worker_id: int):
        """Process a task."""
        print(f"Worker {worker_id} processing task {task.id} ({task.type})")
        
        # Update task status
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.utcnow()
        self._save_tasks()
        
        # Get handler
        handler = self.handlers.get(task.type)
        if not handler:
            raise ValueError(f"No handler registered for task type: {task.type}")
        
        # Execute handler
        result = await handler(task)
        
        # Update task
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.utcnow()
        task.result = result
        task.progress = 100
        self._save_tasks()
        
        print(f"Worker {worker_id} completed task {task.id}")
    
    def stop_worker(self):
        """Stop all workers."""
        self.running = False
        print("Stopping workers...")


# Global task queue instance
_task_queue: Optional[TaskQueue] = None


def get_task_queue() -> TaskQueue:
    """Get global task queue instance."""
    global _task_queue
    if _task_queue is None:
        _task_queue = TaskQueue()
    return _task_queue
