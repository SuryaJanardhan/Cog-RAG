"""
Rate limiting middleware for API protection.
"""

import time
from typing import Optional
from collections import defaultdict
from datetime import datetime, timedelta
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from src.tenants.models import APIKey


class RateLimiter:
    """
    Simple in-memory rate limiter.
    
    In production, use Redis for distributed rate limiting.
    """
    
    def __init__(self):
        """Initialize rate limiter."""
        # Structure: {key: [(timestamp, count)]}
        self.requests: dict[str, list[tuple[float, int]]] = defaultdict(list)
    
    def is_allowed(
        self,
        key: str,
        limit: int,
        window_seconds: int,
    ) -> tuple[bool, Optional[int]]:
        """
        Check if request is allowed under rate limit.
        
        Args:
            key: Rate limit key (e.g., API key or IP)
            limit: Maximum requests allowed
            window_seconds: Time window in seconds
            
        Returns:
            Tuple of (is_allowed, retry_after_seconds)
        """
        now = time.time()
        window_start = now - window_seconds
        
        # Clean old entries
        self.requests[key] = [
            (ts, count) for ts, count in self.requests[key]
            if ts > window_start
        ]
        
        # Count requests in window
        total_requests = sum(count for _, count in self.requests[key])
        
        if total_requests >= limit:
            # Calculate retry after
            oldest = min(ts for ts, _ in self.requests[key]) if self.requests[key] else now
            retry_after = int(oldest + window_seconds - now) + 1
            return False, retry_after
        
        # Allow request
        self.requests[key].append((now, 1))
        return True, None
    
    def get_usage(self, key: str, window_seconds: int) -> dict:
        """Get current usage stats for a key."""
        now = time.time()
        window_start = now - window_seconds
        
        requests_in_window = [
            (ts, count) for ts, count in self.requests.get(key, [])
            if ts > window_start
        ]
        
        total = sum(count for _, count in requests_in_window)
        
        return {
            "requests": total,
            "window_seconds": window_seconds,
            "reset_at": int(now + window_seconds) if requests_in_window else 0,
        }


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting.
    
    Applies different limits based on API key configuration.
    """
    
    def __init__(self, app, limiter: Optional[RateLimiter] = None):
        """
        Initialize middleware.
        
        Args:
            app: FastAPI app
            limiter: RateLimiter instance
        """
        super().__init__(app)
        self.limiter = limiter or RateLimiter()
    
    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting."""
        
        # Skip rate limiting for health check and docs
        if request.url.path in ["/health", "/docs", "/openapi.json", "/redoc"]:
            return await call_next(request)
        
        # Get API key from state (set by auth middleware)
        api_key: Optional[APIKey] = getattr(request.state, "api_key", None)
        
        if api_key:
            # Check minute limit
            minute_key = f"{api_key.id}:minute"
            allowed, retry_after = self.limiter.is_allowed(
                minute_key,
                api_key.rate_limit_per_minute,
                60,
            )
            
            if not allowed:
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded. Try again in {retry_after} seconds.",
                    headers={"Retry-After": str(retry_after)},
                )
            
            # Check hour limit
            hour_key = f"{api_key.id}:hour"
            allowed, retry_after = self.limiter.is_allowed(
                hour_key,
                api_key.rate_limit_per_hour,
                3600,
            )
            
            if not allowed:
                raise HTTPException(
                    status_code=429,
                    detail=f"Hourly rate limit exceeded. Try again in {retry_after} seconds.",
                    headers={"Retry-After": str(retry_after)},
                )
            
            # Check day limit
            day_key = f"{api_key.id}:day"
            allowed, retry_after = self.limiter.is_allowed(
                day_key,
                api_key.rate_limit_per_day,
                86400,
            )
            
            if not allowed:
                raise HTTPException(
                    status_code=429,
                    detail=f"Daily rate limit exceeded. Try again in {retry_after} seconds.",
                    headers={"Retry-After": str(retry_after)},
                )
            
            # Add rate limit headers to response
            response = await call_next(request)
            
            minute_usage = self.limiter.get_usage(minute_key, 60)
            response.headers["X-RateLimit-Limit-Minute"] = str(api_key.rate_limit_per_minute)
            response.headers["X-RateLimit-Remaining-Minute"] = str(
                max(0, api_key.rate_limit_per_minute - minute_usage["requests"])
            )
            
            hour_usage = self.limiter.get_usage(hour_key, 3600)
            response.headers["X-RateLimit-Limit-Hour"] = str(api_key.rate_limit_per_hour)
            response.headers["X-RateLimit-Remaining-Hour"] = str(
                max(0, api_key.rate_limit_per_hour - hour_usage["requests"])
            )
            
            return response
        
        # No API key, apply default IP-based limiting
        client_ip = request.client.host if request.client else "unknown"
        ip_key = f"ip:{client_ip}:minute"
        
        allowed, retry_after = self.limiter.is_allowed(
            ip_key,
            limit=30,  # Default: 30 requests per minute for unauthenticated
            window_seconds=60,
        )
        
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Try again in {retry_after} seconds.",
                headers={"Retry-After": str(retry_after)},
            )
        
        response = await call_next(request)
        return response


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter
