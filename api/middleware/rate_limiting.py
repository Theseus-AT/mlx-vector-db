# api/middleware/rate_limiting.py
"""
Production-Grade Rate Limiting für MLX Vector Database
Intelligente Rate Limits basierend auf Operation Type und User Tier
"""

import time
import asyncio
import logging
import os
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import threading
import redis.asyncio as redis
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
import json

logger = logging.getLogger("mlx_rate_limiter")

class UserTier(Enum):
    """User Tier für differentielle Rate Limits"""
    FREE = "free"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"

class OperationType(Enum):
    """Operation Types mit verschiedenen Rate Limits"""
    QUERY = "query"
    ADD_VECTORS = "add_vectors"
    BATCH_QUERY = "batch_query"
    ADMIN = "admin"
    HEALTH = "health"

@dataclass
class RateLimitConfig:
    """Rate Limit Configuration pro User Tier und Operation"""
    requests_per_minute: int
    requests_per_hour: int
    burst_allowance: int = 0
    concurrent_requests: int = 10

# Production Rate Limit Policies
RATE_LIMIT_POLICIES = {
    UserTier.FREE: {
        OperationType.QUERY: RateLimitConfig(100, 1000, burst_allowance=20),
        OperationType.ADD_VECTORS: RateLimitConfig(20, 200, burst_allowance=5),
        OperationType.BATCH_QUERY: RateLimitConfig(10, 100, burst_allowance=2),
        OperationType.ADMIN: RateLimitConfig(5, 50),
        OperationType.HEALTH: RateLimitConfig(60, 600, burst_allowance=10)
    },
    UserTier.PREMIUM: {
        OperationType.QUERY: RateLimitConfig(1000, 10000, burst_allowance=200),
        OperationType.ADD_VECTORS: RateLimitConfig(200, 2000, burst_allowance=50),
        OperationType.BATCH_QUERY: RateLimitConfig(100, 1000, burst_allowance=20),
        OperationType.ADMIN: RateLimitConfig(50, 500, burst_allowance=10),
        OperationType.HEALTH: RateLimitConfig(300, 3000, burst_allowance=50)
    },
    UserTier.ENTERPRISE: {
        OperationType.QUERY: RateLimitConfig(5000, 50000, burst_allowance=1000),
        OperationType.ADD_VECTORS: RateLimitConfig(1000, 10000, burst_allowance=200),
        OperationType.BATCH_QUERY: RateLimitConfig(500, 5000, burst_allowance=100),
        OperationType.ADMIN: RateLimitConfig(200, 2000, burst_allowance=50),
        OperationType.HEALTH: RateLimitConfig(1000, 10000, burst_allowance=200)
    }
}

class InMemoryRateLimiter:
    """In-Memory Rate Limiter für Single-Instance Deployments"""
    
    def __init__(self):
        self._windows: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self._requests: Dict[str, deque] = defaultdict(deque)
        self._concurrent: Dict[str, int] = defaultdict(int)
        self._lock = threading.RLock()
    
    async def check_rate_limit(self, key: str, config: RateLimitConfig) -> Dict[str, Any]:
        """Check rate limit für given key"""
        current_time = int(time.time())
        current_minute = current_time // 60
        current_hour = current_time // 3600
        
        with self._lock:
            # Clean old requests
            minute_requests = self._windows[f"{key}:minute"]
            hour_requests = self._windows[f"{key}:hour"]
            
            # Remove old windows
            for minute in list(minute_requests.keys()):
                if minute < current_minute - 1:
                    del minute_requests[minute]
            
            for hour in list(hour_requests.keys()):
                if hour < current_hour - 1:
                    del hour_requests[hour]
            
            # Count current requests
            minute_count = minute_requests.get(current_minute, 0)
            hour_count = hour_requests.get(current_hour, 0)
            concurrent_count = self._concurrent[key]
            
            # Check limits
            if minute_count >= config.requests_per_minute:
                return {
                    "allowed": False,
                    "reason": "minute_limit_exceeded",
                    "reset_time": (current_minute + 1) * 60,
                    "current_usage": minute_count,
                    "limit": config.requests_per_minute
                }
            
            if hour_count >= config.requests_per_hour:
                return {
                    "allowed": False,
                    "reason": "hour_limit_exceeded", 
                    "reset_time": (current_hour + 1) * 3600,
                    "current_usage": hour_count,
                    "limit": config.requests_per_hour
                }
            
            if concurrent_count >= config.concurrent_requests:
                return {
                    "allowed": False,
                    "reason": "concurrent_limit_exceeded",
                    "current_usage": concurrent_count,
                    "limit": config.concurrent_requests
                }
            
            # Allow request and increment counters
            minute_requests[current_minute] += 1
            hour_requests[current_hour] += 1
            self._concurrent[key] += 1
            
            return {
                "allowed": True,
                "minute_remaining": config.requests_per_minute - minute_count - 1,
                "hour_remaining": config.requests_per_hour - hour_count - 1,
                "concurrent_usage": concurrent_count + 1
            }
    
    async def release_concurrent(self, key: str):
        """Release concurrent request slot"""
        with self._lock:
            if self._concurrent[key] > 0:
                self._concurrent[key] -= 1

class RedisRateLimiter:
    """Redis-based Rate Limiter für Multi-Instance Deployments"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self._redis = None
    
    async def _get_redis(self):
        """Lazy Redis connection"""
        if self._redis is None:
            self._redis = redis.from_url(self.redis_url)
        return self._redis
    
    async def check_rate_limit(self, key: str, config: RateLimitConfig) -> Dict[str, Any]:
        """Check rate limit mit Redis Lua Script für Atomicity"""
        r = await self._get_redis()
        current_time = int(time.time())
        
        # Lua script für atomic rate limiting
        lua_script = """
        local key = KEYS[1]
        local minute_limit = tonumber(ARGV[1])
        local hour_limit = tonumber(ARGV[2])
        local concurrent_limit = tonumber(ARGV[3])
        local current_time = tonumber(ARGV[4])
        local current_minute = math.floor(current_time / 60)
        local current_hour = math.floor(current_time / 3600)
        
        -- Keys für different windows
        local minute_key = key .. ":minute:" .. current_minute
        local hour_key = key .. ":hour:" .. current_hour
        local concurrent_key = key .. ":concurrent"
        
        -- Get current counts
        local minute_count = tonumber(redis.call('GET', minute_key) or '0')
        local hour_count = tonumber(redis.call('GET', hour_key) or '0')
        local concurrent_count = tonumber(redis.call('GET', concurrent_key) or '0')
        
        -- Check limits
        if minute_count >= minute_limit then
            return {0, 'minute_limit_exceeded', minute_count, minute_limit}
        end
        
        if hour_count >= hour_limit then
            return {0, 'hour_limit_exceeded', hour_count, hour_limit}
        end
        
        if concurrent_count >= concurrent_limit then
            return {0, 'concurrent_limit_exceeded', concurrent_count, concurrent_limit}
        end
        
        -- Increment counters
        redis.call('INCR', minute_key)
        redis.call('EXPIRE', minute_key, 60)
        redis.call('INCR', hour_key)
        redis.call('EXPIRE', hour_key, 3600)
        redis.call('INCR', concurrent_key)
        redis.call('EXPIRE', concurrent_key, 300)  -- 5 minute expiry für safety
        
        return {1, 'allowed', minute_count + 1, hour_count + 1, concurrent_count + 1}
        """
        
        try:
            result = await r.eval(
                lua_script, 
                1, 
                key,
                config.requests_per_minute,
                config.requests_per_hour, 
                config.concurrent_requests,
                current_time
            )
            
            if result[0] == 1:  # Allowed
                return {
                    "allowed": True,
                    "minute_usage": result[2],
                    "hour_usage": result[3],
                    "concurrent_usage": result[4]
                }
            else:  # Denied
                return {
                    "allowed": False,
                    "reason": result[1],
                    "current_usage": result[2],
                    "limit": result[3]
                }
                
        except Exception as e:
            logger.error(f"Redis rate limiting error: {e}")
            # Fall back to allowing request on Redis errors
            return {"allowed": True, "fallback": True}
    
    async def release_concurrent(self, key: str):
        """Release concurrent request slot"""
        try:
            r = await self._get_redis()
            concurrent_key = f"{key}:concurrent"
            await r.decr(concurrent_key)
        except Exception as e:
            logger.error(f"Error releasing concurrent slot: {e}")

class ProductionRateLimiter:
    """Production Rate Limiter mit automatischem Fallback"""
    
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_limiter = RedisRateLimiter(redis_url) if redis_url else None
        self.memory_limiter = InMemoryRateLimiter()
        
    async def check_rate_limit(self, user_id: str, user_tier: UserTier, 
                             operation: OperationType) -> Dict[str, Any]:
        """Check rate limit mit automatic fallback"""
        
        # Get rate limit config
        config = RATE_LIMIT_POLICIES[user_tier][operation]
        rate_limit_key = f"rate_limit:{user_id}:{operation.value}"
        
        # Try Redis first, fall back to memory
        if self.redis_limiter:
            try:
                result = await self.redis_limiter.check_rate_limit(rate_limit_key, config)
                result["backend"] = "redis"
                return result
            except Exception as e:
                logger.warning(f"Redis rate limiter failed, falling back to memory: {e}")
        
        result = await self.memory_limiter.check_rate_limit(rate_limit_key, config)
        result["backend"] = "memory"
        return result
    
    async def release_concurrent(self, user_id: str, operation: OperationType):
        """Release concurrent request slot"""
        rate_limit_key = f"rate_limit:{user_id}:{operation.value}"
        
        if self.redis_limiter:
            try:
                await self.redis_limiter.release_concurrent(rate_limit_key)
                return
            except Exception:
                pass
        
        await self.memory_limiter.release_concurrent(rate_limit_key)

# Global rate limiter instance
rate_limiter = ProductionRateLimiter(
    redis_url=os.getenv("REDIS_URL")
)

def get_user_tier(request: Request) -> UserTier:
    """Determine user tier from request"""
    # Extract from JWT token or API key
    auth_header = request.headers.get("authorization", "")
    
    if "enterprise" in auth_header.lower():
        return UserTier.ENTERPRISE
    elif "premium" in auth_header.lower():
        return UserTier.PREMIUM
    else:
        return UserTier.FREE

def get_operation_type(request: Request) -> OperationType:
    """Determine operation type from request path"""
    path = request.url.path
    
    if "/query" in path:
        return OperationType.BATCH_QUERY if "batch" in path else OperationType.QUERY
    elif "/add" in path:
        return OperationType.ADD_VECTORS
    elif "/admin" in path:
        return OperationType.ADMIN
    elif "/health" in path or "/monitoring" in path:
        return OperationType.HEALTH
    else:
        return OperationType.QUERY  # Default

async def rate_limit_middleware(request: Request, call_next: Callable) -> Response:
    """FastAPI Middleware für Rate Limiting"""
    
    # Skip rate limiting für health checks von Load Balancers
    if request.url.path in ["/health", "/"] and "load-balancer" in request.headers.get("user-agent", "").lower():
        return await call_next(request)
    
    # Get user info from request
    user_id = getattr(request.state, "user_id", request.client.host)  # Fallback to IP
    user_tier = get_user_tier(request)
    operation = get_operation_type(request)
    
    # Check rate limit
    rate_check = await rate_limiter.check_rate_limit(user_id, user_tier, operation)
    
    if not rate_check["allowed"]:
        # Add rate limit headers
        headers = {
            "X-RateLimit-Limit": str(rate_check.get("limit", 0)),
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": str(rate_check.get("reset_time", int(time.time()) + 60)),
            "Retry-After": "60"
        }
        
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "reason": rate_check.get("reason", "unknown"),
                "reset_time": rate_check.get("reset_time"),
                "user_tier": user_tier.value,
                "operation": operation.value
            },
            headers=headers
        )
    
    # Process request
    try:
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(RATE_LIMIT_POLICIES[user_tier][operation].requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(rate_check.get("minute_remaining", 0))
        response.headers["X-RateLimit-Backend"] = rate_check.get("backend", "memory")
        
        return response
        
    finally:
        # Release concurrent slot
        await rate_limiter.release_concurrent(user_id, operation)