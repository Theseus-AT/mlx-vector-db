# sdk/python/mlx_vector_client.py
"""
Production-Ready Python SDK für MLX Vector Database
Async/Await Support + Connection Pooling + Auto-Retry + Type Hints

🎯 Developer Experience:
- Ein-Liner für häufige Operationen
- Intelligent Connection Management
- Automatic Error Recovery
- Full Type Safety
- Context Manager Support
"""

import asyncio
import httpx
import json
import time
import logging
from typing import (
    List, Dict, Any, Optional, Union, AsyncGenerator, AsyncContextManager,
    Tuple, Callable, TypeVar, Generic, overload
)
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from functools import wraps
import numpy as np
from pathlib import Path
import tempfile
import aiofiles
from urllib.parse import urljoin

# MLX support (optional)
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

logger = logging.getLogger("mlx_vector_client")

# =================== TYPE DEFINITIONS ===================

VectorType = Union[List[float], np.ndarray]
if MLX_AVAILABLE:
    VectorType = Union[List[float], np.ndarray, mx.array]

MetadataType = Dict[str, Any]
QueryResultType = Dict[str, Any]

T = TypeVar('T')

@dataclass
class ClientConfig:
    """Configuration for MLX Vector Client"""
    base_url: str = "http://localhost:8000"
    api_key: Optional[str] = None
    jwt_token: Optional[str] = None
    
    # Connection settings
    timeout: float = 30.0
    max_connections: int = 100
    max_keepalive_connections: int = 20
    keepalive_expiry: float = 30.0
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_exponential_base: float = 2.0
    retry_jitter: bool = True
    
    # Batch settings
    default_batch_size: int = 1000
    max_batch_size: int = 10000
    
    # Streaming settings
    enable_streaming: bool = True
    stream_chunk_size: int = 8192
    
    # Performance settings
    enable_compression: bool = True
    enable_http2: bool = True
    enable_connection_pooling: bool = True

@dataclass 
class VectorSearchResult:
    """Structured result for vector search"""
    index: int
    distance: float
    similarity_score: float
    metadata: MetadataType
    rank: int

@dataclass
class BatchOperationResult:
    """Result for batch operations"""
    operation_id: str
    success: bool
    total_processed: int
    total_errors: int
    processing_time_ms: float
    throughput_items_per_sec: float
    errors: List[str] = field(default_factory=list)

@dataclass
class StreamProgress:
    """Progress update for streaming operations"""
    operation_id: str
    progress_percent: float
    items_processed: int
    total_items: int
    elapsed_time_ms: float
    estimated_remaining_ms: Optional[float] = None

# =================== CONNECTION MANAGER ===================

class ConnectionManager:
    """Intelligent connection pool manager"""
    
    def __init__(self, config: ClientConfig):
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None
        self._lock = asyncio.Lock()
        
        # Connection stats
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "retry_attempts": 0,
            "avg_response_time_ms": 0.0
        }
    
    async def get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with connection pooling"""
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    # Build headers
                    headers = {
                        "User-Agent": "MLX-VectorDB-SDK/1.0",
                        "Content-Type": "application/json"
                    }
                    
                    if self.config.enable_compression:
                        headers["Accept-Encoding"] = "gzip, deflate"
                    
                    # Authentication
                    if self.config.jwt_token:
                        headers["Authorization"] = f"Bearer {self.config.jwt_token}"
                    elif self.config.api_key:
                        headers["Authorization"] = f"Bearer {self.config.api_key}"
                    
                    # Connection limits
                    limits = httpx.Limits(
                        max_connections=self.config.max_connections,
                        max_keepalive_connections=self.config.max_keepalive_connections,
                        keepalive_expiry=self.config.keepalive_expiry
                    )
                    
                    # Create client
                    self._client = httpx.AsyncClient(
                        base_url=self.config.base_url,
                        headers=headers,
                        timeout=httpx.Timeout(self.config.timeout),
                        limits=limits,
                        http2=self.config.enable_http2
                    )
        
        return self._client
    
    async def close(self):
        """Close connection pool"""
        if self._client:
            await self._client.aclose()
            self._client = None

# =================== RETRY DECORATOR ===================

def with_retry(max_retries: int = None, delay: float = None):
    """Decorator for automatic retry with exponential backoff"""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            max_attempts = max_retries or self.config.max_retries
            base_delay = delay or self.config.retry_delay
            
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    result = await func(self, *args, **kwargs)
                    
                    # Record successful request
                    self.connection_manager.stats["successful_requests"] += 1
                    
                    return result
                
                except httpx.HTTPStatusError as e:
                    # Don't retry client errors (4xx)
                    if 400 <= e.response.status_code < 500:
                        self.connection_manager.stats["failed_requests"] += 1
                        raise
                    
                    last_exception = e
                    
                except (httpx.RequestError, httpx.TimeoutException) as e:
                    last_exception = e
                
                # Calculate delay with exponential backoff
                if attempt < max_attempts - 1:
                    retry_delay = base_delay * (self.config.retry_exponential_base ** attempt)
                    
                    # Add jitter
                    if self.config.retry_jitter:
                        import random
                        retry_delay *= (0.5 + random.random() * 0.5)
                    
                    logger.warning(f"Request failed, retrying in {retry_delay:.2f}s (attempt {attempt + 1}/{max_attempts})")
                    
                    self.connection_manager.stats["retry_attempts"] += 1
                    await asyncio.sleep(retry_delay)
            
            # All retries exhausted
            self.connection_manager.stats["failed_requests"] += 1
            raise last_exception
            
        return wrapper
    return decorator

# =================== MAIN CLIENT CLASS ===================

class MLXVectorClient:
    """Production-ready MLX Vector Database Client"""
    
    def __init__(self, config: Optional[ClientConfig] = None, **kwargs):
        """Initialize client with configuration"""
        
        if config is None:
            config = ClientConfig(**kwargs)
        
        self.config = config
        self.connection_manager = ConnectionManager(config)
        
        # Request tracking
        self._request_times = []
        self._start_time = time.time()
        
        logger.info(f"MLX Vector Client initialized: {config.base_url}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    # =================== UTILITY METHODS ===================
    
    def _convert_vector(self, vector: VectorType) -> List[float]:
        """Convert vector to JSON-serializable format"""
        if isinstance(vector, list):
            return vector
        elif isinstance(vector, np.ndarray):
            return vector.astype(np.float32).tolist()
        elif MLX_AVAILABLE and isinstance(vector, mx.array):
            return vector.tolist()
        else:
            raise TypeError(f"Unsupported vector type: {type(vector)}")
    
    def _convert_vectors_batch(self, vectors: List[VectorType]) -> List[List[float]]:
        """Convert batch of vectors to JSON-serializable format"""
        return [self._convert_vector(v) for v in vectors]
    
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request with timing and error handling"""
        
        start_time = time.time()
        
        try:
            client = await self.connection_manager.get_client()
            response = await client.request(method, endpoint, **kwargs)
            
            # Check for errors
            response.raise_for_status()
            
            # Parse response
            result = response.json() if response.content else {}
            
            # Track timing
            request_time = (time.time() - start_time) * 1000
            self._request_times.append(request_time)
            
            # Keep only recent times for average calculation
            if len(self._request_times) > 100:
                self._request_times.pop(0)
            
            # Update average response time
            self.connection_manager.stats["avg_response_time_ms"] = sum(self._request_times) / len(self._request_times)
            self.connection_manager.stats["total_requests"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Request failed: {method} {endpoint} - {e}")
            raise
    
    # =================== STORE MANAGEMENT ===================
    
    @with_retry()
    async def create_store(self, user_id: str, model_id: str, **config_kwargs) -> Dict[str, Any]:
        """Create a new vector store"""
        
        payload = {
            "user_id": user_id,
            "model_id": model_id,
            **config_kwargs
        }
        
        return await self._make_request("POST", "/admin/create_store", json=payload)
    
    @with_retry()
    async def delete_store(self, user_id: str, model_id: str, force: bool = False) -> Dict[str, Any]:
        """Delete a vector store"""
        
        payload = {
            "user_id": user_id,
            "model_id": model_id,
            "force": force
        }
        
        return await self._make_request("DELETE", "/admin/store", json=payload)
    
    @with_retry()
    async def get_store_stats(self, user_id: str, model_id: str) -> Dict[str, Any]:
        """Get statistics for a vector store"""
        
        params = {"user_id": user_id, "model_id": model_id}
        return await self._make_request("GET", "/admin/store/stats", params=params)
    
    @with_retry()
    async def list_stores(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """List all stores (optionally filtered by user)"""
        
        params = {"user_id": user_id} if user_id else {}
        return await self._make_request("GET", "/admin/list_stores", params=params)
    
    # =================== VECTOR OPERATIONS ===================
    
    @with_retry()
    async def add_vectors(self, user_id: str, model_id: str, 
                         vectors: List[VectorType], 
                         metadata: List[MetadataType]) -> Dict[str, Any]:
        """Add vectors to store"""
        
        if len(vectors) != len(metadata):
            raise ValueError("Vectors and metadata must have the same length")
        
        payload = {
            "user_id": user_id,
            "model_id": model_id,
            "vectors": self._convert_vectors_batch(vectors),
            "metadata": metadata
        }
        
        return await self._make_request("POST", "/vectors/add", json=payload)
    
    @with_retry()
    async def query_vectors(self, user_id: str, model_id: str,
                           query_vector: VectorType, k: int = 10,
                           filter_metadata: Optional[MetadataType] = None) -> List[VectorSearchResult]:
        """Query similar vectors"""
        
        payload = {
            "user_id": user_id,
            "model_id": model_id,
            "query": self._convert_vector(query_vector),
            "k": k
        }
        
        if filter_metadata:
            payload["filter_metadata"] = filter_metadata
        
        response = await self._make_request("POST", "/vectors/query", json=payload)
        
        # Convert to structured results
        results = []
        for result in response.get("results", []):
            results.append(VectorSearchResult(
                index=result.get("index", -1),
                distance=result.get("distance", 0.0),
                similarity_score=result.get("similarity_score", 0.0),
                metadata=result.get("metadata", {}),
                rank=result.get("rank", 0)
            ))
        
        return results
    
    @with_retry()
    async def batch_query(self, user_id: str, model_id: str,
                         query_vectors: List[VectorType], k: int = 10) -> List[List[VectorSearchResult]]:
        """Batch query multiple vectors"""
        
        payload = {
            "user_id": user_id,
            "model_id": model_id,
            "queries": self._convert_vectors_batch(query_vectors),
            "k": k
        }
        
        response = await self._make_request("POST", "/vectors/batch_query", json=payload)
        
        # Convert to structured results
        batch_results = []
        for query_results in response.get("results", []):
            query_result_objects = []
            for result in query_results:
                query_result_objects.append(VectorSearchResult(
                    index=result.get("index", -1),
                    distance=result.get("distance", 0.0), 
                    similarity_score=result.get("similarity_score", 0.0),
                    metadata=result.get("metadata", {}),
                    rank=result.get("rank", 0)
                ))
            batch_results.append(query_result_objects)
        
        return batch_results
    
    @with_retry()
    async def count_vectors(self, user_id: str, model_id: str) -> int:
        """Count vectors in store"""
        
        params = {"user_id": user_id, "model_id": model_id}
        response = await self._make_request("GET", "/vectors/count", params=params)
        
        return response.get("count", 0)
    
    @with_retry()
    async def delete_vectors(self, user_id: str, model_id: str,
                           filter_metadata: MetadataType) -> int:
        """Delete vectors by metadata filter"""
        
        payload = {
            "user_id": user_id,
            "model_id": model_id,
            "filter_metadata": filter_metadata
        }
        
        response = await self._make_request("POST", "/vectors/delete", json=payload)
        return response.get("deleted", 0)
    
    # =================== BATCH OPERATIONS ===================
    
    @with_retry()
    async def batch_add_vectors(self, user_id: str, model_id: str,
                               vectors: List[VectorType],
                               metadata: List[MetadataType],
                               batch_size: Optional[int] = None,
                               enable_streaming: bool = False) -> Union[Dict[str, Any], BatchOperationResult]:
        """Batch add vectors with optional streaming"""
        
        batch_size = batch_size or self.config.default_batch_size
        
        payload = {
            "user_id": user_id,
            "model_id": model_id,
            "vectors": self._convert_vectors_batch(vectors),
            "metadata": metadata,
            "batch_size": batch_size,
            "enable_streaming": enable_streaming
        }
        
        response = await self._make_request("POST", "/v1/batch/vectors/add", json=payload)
        
        if enable_streaming:
            return response  # Contains operation_id and stream URLs
        else:
            return BatchOperationResult(
                operation_id=response.get("operation_id", ""),
                success=response.get("success", False),
                total_processed=response.get("total_processed", 0),
                total_errors=response.get("total_errors", 0),
                processing_time_ms=response.get("processing_time_ms", 0),
                throughput_items_per_sec=response.get("throughput_items_per_sec", 0),
                errors=response.get("errors", [])
            )
    
    async def stream_batch_progress(self, operation_id: str) -> AsyncGenerator[StreamProgress, None]:
        """Stream progress updates for batch operations"""
        
        url = f"/v1/batch/stream/{operation_id}"
        
        client = await self.connection_manager.get_client()
        
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])  # Remove "data: " prefix
                        
                        if "operation_id" in data:
                            yield StreamProgress(
                                operation_id=data["operation_id"],
                                progress_percent=data.get("progress_percent", 0),
                                items_processed=data.get("items_processed", 0),
                                total_items=data.get("total_items", 0),
                                elapsed_time_ms=data.get("elapsed_time_ms", 0),
                                estimated_remaining_ms=data.get("estimated_remaining_ms")
                            )
                        
                        # Check for completion
                        if data.get("progress_percent", 0) >= 100:
                            break
                            
                    except json.JSONDecodeError:
                        continue  # Skip invalid JSON lines
    
    @with_retry()
    async def upload_file(self, user_id: str, model_id: str,
                         file_path: Union[str, Path],
                         file_format: str = "npz",
                         chunk_size: int = 5000) -> Dict[str, Any]:
        """Upload vectors from file"""
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Prepare multipart upload
        async with aiofiles.open(file_path, 'rb') as f:
            file_content = await f.read()
        
        # Request payload
        data = {
            "user_id": user_id,
            "model_id": model_id,
            "file_format": file_format,
            "chunk_size": chunk_size
        }
        
        files = {"file": (file_path.name, file_content, "application/octet-stream")}
        
        client = await self.connection_manager.get_client()
        response = await client.post("/v1/batch/upload", data=data, files=files)
        response.raise_for_status()
        
        return response.json()
    
    # =================== HEALTH & MONITORING ===================
    
    @with_retry()
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        return await self._make_request("GET", "/health")
    
    @with_retry()
    async def performance_health(self) -> Dict[str, Any]:
        """Get performance health status"""
        return await self._make_request("GET", "/performance/health")
    
    @with_retry()
    async def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return await self._make_request("GET", "/system/info")
    
    # =================== CONTEXT MANAGERS ===================
    
    @asynccontextmanager
    async def store_context(self, user_id: str, model_id: str, 
                           auto_create: bool = True) -> AsyncContextManager["StoreContext"]:
        """Context manager for store operations"""
        
        store_context = StoreContext(self, user_id, model_id)
        
        if auto_create:
            try:
                await self.create_store(user_id, model_id)
            except Exception:
                pass  # Store might already exist
        
        try:
            yield store_context
        finally:
            pass  # Cleanup if needed
    
    # =================== ONE-LINER CONVENIENCE METHODS ===================
    
    async def quick_add(self, user_id: str, model_id: str,
                       texts: List[str], embeddings: List[VectorType]) -> Dict[str, Any]:
        """One-liner: Add text embeddings to store"""
        
        if len(texts) != len(embeddings):
            raise ValueError("Texts and embeddings must have same length")
        
        metadata = [{"text": text, "timestamp": time.time()} for text in texts]
        
        # Auto-create store if needed
        try:
            await self.create_store(user_id, model_id)
        except Exception:
            pass  # Store might already exist
        
        return await self.add_vectors(user_id, model_id, embeddings, metadata)
    
    async def quick_search(self, user_id: str, model_id: str,
                          query_embedding: VectorType, k: int = 5) -> List[str]:
        """One-liner: Search and return text results"""
        
        results = await self.query_vectors(user_id, model_id, query_embedding, k)
        
        return [result.metadata.get("text", "") for result in results]
    
    async def semantic_search(self, user_id: str, model_id: str,
                             query_text: str, embedding_func: Callable[[str], VectorType],
                             k: int = 5) -> List[Dict[str, Any]]:
        """One-liner: Semantic search with automatic embedding"""
        
        query_embedding = embedding_func(query_text)
        results = await self.query_vectors(user_id, model_id, query_embedding, k)
        
        return [
            {
                "text": result.metadata.get("text", ""),
                "similarity": result.similarity_score,
                "metadata": result.metadata
            }
            for result in results
        ]
    
    # =================== PERFORMANCE & STATS ===================
    
    def get_client_stats(self) -> Dict[str, Any]:
        """Get client performance statistics"""
        
        uptime = time.time() - self._start_time
        
        return {
            **self.connection_manager.stats,
            "uptime_seconds": uptime,
            "config": {
                "base_url": self.config.base_url,
                "max_retries": self.config.max_retries,
                "timeout": self.config.timeout,
                "enable_streaming": self.config.enable_streaming
            }
        }
    
    async def benchmark_performance(self, user_id: str, model_id: str,
                                  num_vectors: int = 1000,
                                  dimension: int = 384) -> Dict[str, Any]:
        """Benchmark client performance"""
        
        # Generate test data
        test_vectors = np.random.rand(num_vectors, dimension).astype(np.float32)
        test_metadata = [{"id": f"benchmark_{i}"} for i in range(num_vectors)]
        
        # Benchmark add vectors
        start_time = time.time()
        add_result = await self.add_vectors(user_id, model_id, test_vectors, test_metadata)
        add_time = time.time() - start_time
        
        # Benchmark queries
        query_times = []
        for i in range(min(100, num_vectors)):
            query_start = time.time()
            await self.query_vectors(user_id, model_id, test_vectors[i], k=10)
            query_times.append(time.time() - query_start)
        
        avg_query_time = sum(query_times) / len(query_times)
        
        return {
            "add_performance": {
                "vectors_added": num_vectors,
                "time_seconds": add_time,
                "vectors_per_second": num_vectors / add_time
            },
            "query_performance": {
                "queries_tested": len(query_times),
                "avg_time_seconds": avg_query_time,
                "queries_per_second": 1 / avg_query_time
            },
            "client_stats": self.get_client_stats()
        }
    
    # =================== CLEANUP ===================
    
    async def close(self):
        """Close client and cleanup resources"""
        await self.connection_manager.close()
        logger.info("MLX Vector Client closed")

# =================== STORE CONTEXT HELPER ===================

class StoreContext:
    """Context helper for store-specific operations"""
    
    def __init__(self, client: MLXVectorClient, user_id: str, model_id: str):
        self.client = client
        self.user_id = user_id
        self.model_id = model_id
    
    async def add(self, vectors: List[VectorType], metadata: List[MetadataType]) -> Dict[str, Any]:
        """Add vectors to this store"""
        return await self.client.add_vectors(self.user_id, self.model_id, vectors, metadata)
    
    async def query(self, vector: VectorType, k: int = 10, 
                   filter_metadata: Optional[MetadataType] = None) -> List[VectorSearchResult]:
        """Query vectors in this store"""
        return await self.client.query_vectors(self.user_id, self.model_id, vector, k, filter_metadata)
    
    async def count(self) -> int:
        """Count vectors in this store"""
        return await self.client.count_vectors(self.user_id, self.model_id)
    
    async def stats(self) -> Dict[str, Any]:
        """Get stats for this store"""
        return await self.client.get_store_stats(self.user_id, self.model_id)

# =================== FACTORY FUNCTIONS ===================

def create_client(base_url: str = "http://localhost:8000", 
                 api_key: Optional[str] = None,
                 **config_kwargs) -> MLXVectorClient:
    """Factory function to create MLX Vector Client"""
    
    config = ClientConfig(
        base_url=base_url,
        api_key=api_key,
        **config_kwargs
    )
    
    return MLXVectorClient(config)

async def create_async_client(base_url: str = "http://localhost:8000",
                            api_key: Optional[str] = None,
                            **config_kwargs) -> MLXVectorClient:
    """Factory function to create and initialize async client"""
    
    client = create_client(base_url, api_key, **config_kwargs)
    
    # Test connection
    try:
        await client.health_check()
        logger.info("Client successfully connected to MLX Vector Database")
    except Exception as e:
        logger.warning(f"Initial connection test failed: {e}")
    
    return client

# =================== USAGE EXAMPLES ===================

async def demo_sdk_usage():
    """Demonstrate SDK usage patterns"""
    
    print("🚀 MLX Vector Database SDK Demo")
    print("=" * 40)
    
    # Create client
    async with create_client("http://localhost:8000", api_key="your-api-key") as client:
        
        # 1. One-liner semantic search
        print("\n1️⃣ One-liner Operations:")
        
        # Mock embedding function
        def mock_embedding(text: str) -> List[float]:
            return np.random.rand(384).tolist()
        
        # Add some sample data
        texts = ["Machine learning is amazing", "Vector databases are powerful", "Apple Silicon performance"]
        embeddings = [mock_embedding(text) for text in texts]
        
        await client.quick_add("demo_user", "demo_model", texts, embeddings)
        print("   ✅ Added sample texts with embeddings")
        
        # Semantic search
        results = await client.semantic_search(
            "demo_user", "demo_model", 
            "AI and ML topics", mock_embedding, k=2
        )
        
        print(f"   🔍 Found {len(results)} similar texts:")
        for result in results:
            print(f"      '{result['text']}' (similarity: {result['similarity']:.3f})")
        
        # 2. Context manager usage
        print("\n2️⃣ Context Manager:")
        
        async with client.store_context("demo_user", "context_model") as store:
            # Add vectors
            await store.add([np.random.rand(384)], [{"type": "context_demo"}])
            
            # Query
            results = await store.query(np.random.rand(384), k=1)
            print(f"   ✅ Context store has {await store.count()} vectors")
        
        # 3. Streaming operations
        print("\n3️⃣ Streaming Operations:")
        
        # Start batch operation
        large_vectors = [np.random.rand(384) for _ in range(100)]
        large_metadata = [{"batch": "demo", "id": i} for i in range(100)]
        
        batch_response = await client.batch_add_vectors(
            "demo_user", "batch_model",
            large_vectors, large_metadata,
            enable_streaming=True
        )
        
        if "operation_id" in batch_response:
            print(f"   🔄 Started batch operation: {batch_response['operation_id']}")
            
            # Stream progress
            async for progress in client.stream_batch_progress(batch_response["operation_id"]):
                print(f"   📊 Progress: {progress.progress_percent:.1f}% ({progress.items_processed}/{progress.total_items})")
                
                if progress.progress_percent >= 100:
                    break
        
        # 4. Performance benchmark
        print("\n4️⃣ Performance Benchmark:")
        
        perf_results = await client.benchmark_performance(
            "demo_user", "perf_model", num_vectors=500
        )
        
        print(f"   📈 Add Performance: {perf_results['add_performance']['vectors_per_second']:.1f} vectors/sec")
        print(f"   ⚡ Query Performance: {perf_results['query_performance']['queries_per_second']:.1f} queries/sec")
        
        # 5. Client statistics
        print("\n5️⃣ Client Statistics:")
        
        stats = client.get_client_stats()
        print(f"   📊 Total Requests: {stats['total_requests']}")
        print(f"   ✅ Success Rate: {stats['successful_requests'] / max(stats['total_requests'], 1) * 100:.1f}%")
        print(f"   ⏱️ Avg Response Time: {stats['avg_response_time_ms']:.2f}ms")
        
        # Cleanup
        await client.delete_store("demo_user", "demo_model", force=True)
        await client.delete_store("demo_user", "context_model", force=True)
        await client.delete_store("demo_user", "batch_model", force=True)
        await client.delete_store("demo_user", "perf_model", force=True)
        
        print("\n✅ SDK Demo completed successfully!")

if __name__ == "__main__":
    asyncio.run(demo_sdk_usage())