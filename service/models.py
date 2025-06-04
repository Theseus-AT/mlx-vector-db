"""
Storage Models for MLX Vector Database
Optimized Pydantic models for high-performance API requests/responses

Key Features:
- Efficient serialization for large vector data
- Validation for MLX-compatible data types
- Memory-optimized field definitions
- Support for streaming responses
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Union, Any
import numpy as np


class VectorAddRequest(BaseModel):
    """Request model for adding vectors to the store"""
    user_id: str = Field(..., description="User identifier")
    model_id: str = Field(..., description="Model identifier") 
    vectors: List[List[float]] = Field(..., description="List of vector embeddings")
    metadata: List[Dict[str, Any]] = Field(..., description="Metadata for each vector")
    
    @validator('vectors')
    def validate_vectors(cls, v):
        if not v:
            raise ValueError("Vectors list cannot be empty")
        
        # Check all vectors have same dimension
        if len(set(len(vec) for vec in v)) > 1:
            raise ValueError("All vectors must have the same dimension")
        
        # Check for valid float values
        for i, vec in enumerate(v):
            if not all(isinstance(x, (int, float)) for x in vec):
                raise ValueError(f"Vector {i} contains non-numeric values")
        
        return v
    
    @validator('metadata')
    def validate_metadata(cls, v, values):
        if 'vectors' in values and len(v) != len(values['vectors']):
            raise ValueError("Number of metadata entries must match number of vectors")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user123",
                "model_id": "sentence-transformer",
                "vectors": [
                    [0.1, 0.2, 0.3, 0.4],
                    [0.5, 0.6, 0.7, 0.8]
                ],
                "metadata": [
                    {"id": "doc1", "source": "api", "category": "tech"},
                    {"id": "doc2", "source": "api", "category": "tech"}
                ]
            }
        }


class VectorQuery(BaseModel):
    """Request model for vector similarity search"""
    user_id: str = Field(..., description="User identifier")
    model_id: str = Field(..., description="Model identifier")
    query: List[float] = Field(..., description="Query vector for similarity search")
    k: int = Field(default=10, ge=1, le=1000, description="Number of similar vectors to return")
    filter_metadata: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Metadata filters to apply"
    )
    
    @validator('query')
    def validate_query_vector(cls, v):
        if not v:
            raise ValueError("Query vector cannot be empty")
        
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError("Query vector must contain only numeric values")
        
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user123",
                "model_id": "sentence-transformer", 
                "query": [0.1, 0.2, 0.3, 0.4],
                "k": 5,
                "filter_metadata": {"category": "tech"}
            }
        }


class BatchQueryRequest(BaseModel):
    """Request model for batch vector queries"""
    user_id: str = Field(..., description="User identifier")
    model_id: str = Field(..., description="Model identifier")
    queries: List[List[float]] = Field(..., description="List of query vectors")
    k: int = Field(default=10, ge=1, le=1000, description="Number of results per query")
    
    @validator('queries')
    def validate_queries(cls, v):
        if not v:
            raise ValueError("Queries list cannot be empty")
        
        if len(v) > 100:  # Limit batch size for performance
            raise ValueError("Maximum 100 queries per batch")
        
        # Check all queries have same dimension
        dimensions = set(len(query) for query in v)
        if len(dimensions) > 1:
            raise ValueError("All query vectors must have the same dimension")
        
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user123",
                "model_id": "sentence-transformer",
                "queries": [
                    [0.1, 0.2, 0.3, 0.4],
                    [0.5, 0.6, 0.7, 0.8]
                ],
                "k": 5
            }
        }


class VectorStoreConfig(BaseModel):
    """Configuration model for vector store creation"""
    dimension: int = Field(default=384, ge=1, le=4096, description="Vector dimension")
    metric: str = Field(
        default="cosine", 
        regex="^(cosine|euclidean|dot_product)$",
        description="Similarity metric"
    )
    index_type: str = Field(
        default="flat",
        regex="^(flat|hnsw)$", 
        description="Index type for search"
    )
    chunk_size: int = Field(default=1000, ge=100, le=10000, description="Batch processing chunk size")
    cache_size: int = Field(default=10000, ge=1000, le=100000, description="Vector cache size")
    use_metal: bool = Field(default=True, description="Use Metal GPU acceleration")
    jit_compile: bool = Field(default=True, description="Enable JIT compilation")
    
    class Config:
        schema_extra = {
            "example": {
                "dimension": 384,
                "metric": "cosine",
                "index_type": "flat",
                "chunk_size": 1000,
                "cache_size": 10000,
                "use_metal": True,
                "jit_compile": True
            }
        }


class CreateStoreRequest(BaseModel):
    """Request model for creating a new vector store"""
    user_id: str = Field(..., description="User identifier")
    model_id: str = Field(..., description="Model identifier")
    config: Optional[VectorStoreConfig] = Field(
        default=None,
        description="Store configuration (uses defaults if not provided)"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user123",
                "model_id": "sentence-transformer",
                "config": {
                    "dimension": 384,
                    "metric": "cosine"
                }
            }
        }


class VectorSearchResult(BaseModel):
    """Single search result"""
    metadata: Dict[str, Any] = Field(..., description="Vector metadata")
    similarity_score: float = Field(..., description="Similarity score")
    rank: int = Field(..., description="Result rank (1-based)")
    
    class Config:
        schema_extra = {
            "example": {
                "metadata": {"id": "doc1", "source": "api", "category": "tech"},
                "similarity_score": 0.95,
                "rank": 1
            }
        }


class VectorQueryResponse(BaseModel):
    """Response model for vector similarity search"""
    results: List[VectorSearchResult] = Field(..., description="Search results")
    query_time_ms: float = Field(..., description="Query execution time in milliseconds")
    total_vectors_searched: int = Field(..., description="Total number of vectors in store")
    metadata_filter_applied: bool = Field(..., description="Whether metadata filtering was used")
    
    class Config:
        schema_extra = {
            "example": {
                "results": [
                    {
                        "metadata": {"id": "doc1", "category": "tech"},
                        "similarity_score": 0.95,
                        "rank": 1
                    }
                ],
                "query_time_ms": 2.5,
                "total_vectors_searched": 10000,
                "metadata_filter_applied": True
            }
        }


class BatchQueryResponse(BaseModel):
    """Response model for batch vector queries"""
    results: List[List[VectorSearchResult]] = Field(..., description="Batch search results")
    total_queries: int = Field(..., description="Number of queries processed")
    avg_query_time_ms: float = Field(..., description="Average query time in milliseconds")
    total_processing_time_ms: float = Field(..., description="Total batch processing time")
    
    class Config:
        schema_extra = {
            "example": {
                "results": [
                    [
                        {
                            "metadata": {"id": "doc1"},
                            "similarity_score": 0.95,
                            "rank": 1
                        }
                    ]
                ],
                "total_queries": 2,
                "avg_query_time_ms": 2.5,
                "total_processing_time_ms": 5.0
            }
        }


class VectorAddResponse(BaseModel):
    """Response model for vector addition"""
    success: bool = Field(..., description="Operation success status")
    vectors_added: int = Field(..., description="Number of vectors added")
    total_vectors: int = Field(..., description="Total vectors in store after addition")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    store_stats: Dict[str, Any] = Field(..., description="Updated store statistics")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "vectors_added": 100,
                "total_vectors": 10100,
                "processing_time_ms": 50.2,
                "store_stats": {
                    "vector_count": 10100,
                    "dimension": 384,
                    "memory_usage_mb": 15.6
                }
            }
        }


class StoreStatsResponse(BaseModel):
    """Response model for store statistics"""
    store_stats: Dict[str, Any] = Field(..., description="Basic store statistics")
    performance_metrics: Dict[str, Union[str, float]] = Field(..., description="Performance metrics")
    mlx_info: Dict[str, Any] = Field(..., description="MLX framework information")
    
    class Config:
        schema_extra = {
            "example": {
                "store_stats": {
                    "vector_count": 10000,
                    "dimension": 384,
                    "memory_usage_mb": 15.2
                },
                "performance_metrics": {
                    "expected_qps": 1000,
                    "memory_efficiency": "unified_memory"
                },
                "mlx_info": {
                    "mlx_version": "0.25.2",
                    "device": "gpu",
                    "unified_memory": True
                }
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service health status")
    mlx_optimized: bool = Field(..., description="MLX optimization status") 
    stores_active: int = Field(..., description="Number of active stores")
    total_vectors: int = Field(..., description="Total vectors across all stores")
    memory_usage_mb: float = Field(..., description="Total memory usage")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "mlx_optimized": True,
                "stores_active": 5,
                "total_vectors": 50000,
                "memory_usage_mb": 75.3
            }
        }


class BenchmarkRequest(BaseModel):
    """Request model for performance benchmarking"""
    user_id: str = Field(..., description="User identifier")
    model_id: str = Field(..., description="Model identifier")
    num_vectors: int = Field(default=1000, ge=100, le=10000, description="Number of test vectors")
    num_queries: int = Field(default=100, ge=10, le=1000, description="Number of test queries")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user123",
                "model_id": "sentence-transformer",
                "num_vectors": 1000,
                "num_queries": 100
            }
        }


class BenchmarkResponse(BaseModel):
    """Response model for benchmark results"""
    benchmark_results: Dict[str, float] = Field(..., description="Benchmark metrics")
    mlx_optimized: bool = Field(..., description="MLX optimization status")
    performance_target: str = Field(..., description="Performance target description")
    
    class Config:
        schema_extra = {
            "example": {
                "benchmark_results": {
                    "add_rate": 5000.0,
                    "qps": 1200.0,
                    "avg_latency_ms": 0.83,
                    "memory_mb": 15.2
                },
                "mlx_optimized": True,
                "performance_target": "1000+ QPS"
            }
        }


# Error response models
class ErrorResponse(BaseModel):
    """Standard error response"""
    detail: str = Field(..., description="Error description")
    error_code: Optional[str] = Field(None, description="Specific error code")
    timestamp: Optional[str] = Field(None, description="Error timestamp")


class ValidationErrorDetail(BaseModel):
    """Validation error detail"""
    loc: List[Union[str, int]] = Field(..., description="Error location")
    msg: str = Field(..., description="Error message")
    type: str = Field(..., description="Error type")


class ValidationErrorResponse(BaseModel):
    """Validation error response"""
    detail: List[ValidationErrorDetail] = Field(..., description="Validation errors")


# Export/Import models for admin operations
class ExportRequest(BaseModel):
    """Request model for store export"""
    user_id: str = Field(..., description="User identifier")
    model_id: str = Field(..., description="Model identifier")
    include_metadata: bool = Field(default=True, description="Include metadata in export")
    compression_level: int = Field(default=6, ge=0, le=9, description="ZIP compression level")


class ImportRequest(BaseModel):
    """Request model for store import"""
    user_id: str = Field(..., description="User identifier")
    model_id: str = Field(..., description="Model identifier")
    overwrite_existing: bool = Field(default=False, description="Overwrite existing store")
    validate_vectors: bool = Field(default=True, description="Validate imported vectors")


class ExportResponse(BaseModel):
    """Response model for store export"""
    success: bool = Field(..., description="Export success status")
    export_path: str = Field(..., description="Path to exported file")
    file_size_mb: float = Field(..., description="Export file size in MB")
    vectors_exported: int = Field(..., description="Number of vectors exported")
    export_time_ms: float = Field(..., description="Export processing time")


class ImportResponse(BaseModel):
    """Response model for store import"""
    success: bool = Field(..., description="Import success status")
    vectors_imported: int = Field(..., description="Number of vectors imported")
    metadata_imported: int = Field(..., description="Number of metadata entries imported")
    import_time_ms: float = Field(..., description="Import processing time")
    store_recreated: bool = Field(..., description="Whether store was recreated")


# Streaming response models
class StreamResult(BaseModel):
    """Single streaming result"""
    metadata: Optional[Dict[str, Any]] = Field(None, description="Vector metadata")
    similarity_score: Optional[float] = Field(None, description="Similarity score") 
    rank: Optional[int] = Field(None, description="Result rank")
    status: Optional[str] = Field(None, description="Stream status (e.g., 'complete')")
    error: Optional[str] = Field(None, description="Error message if applicable")


# Advanced query models
class AdvancedQueryRequest(BaseModel):
    """Advanced query with multiple search modes"""
    user_id: str = Field(..., description="User identifier")
    model_id: str = Field(..., description="Model identifier")
    query: List[float] = Field(..., description="Query vector")
    k: int = Field(default=10, ge=1, le=1000, description="Number of results")
    search_mode: str = Field(
        default="similarity", 
        regex="^(similarity|hybrid|rerank)$",
        description="Search mode"
    )
    filter_metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    score_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum similarity score")
    diversify_results: bool = Field(default=False, description="Apply result diversification")


class HybridQueryRequest(BaseModel):
    """Hybrid search combining vector and text queries"""
    user_id: str = Field(..., description="User identifier")
    model_id: str = Field(..., description="Model identifier")
    vector_query: List[float] = Field(..., description="Vector query")
    text_query: Optional[str] = Field(None, description="Text query for hybrid search")
    vector_weight: float = Field(default=0.7, ge=0.0, le=1.0, description="Vector search weight")
    text_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="Text search weight")
    k: int = Field(default=10, ge=1, le=1000, description="Number of results")
    
    @validator('text_weight')
    def validate_weights_sum(cls, v, values):
        if 'vector_weight' in values:
            if abs(values['vector_weight'] + v - 1.0) > 0.001:
                raise ValueError("vector_weight + text_weight must equal 1.0")
        return v


# Monitoring and metrics models
class PerformanceMetrics(BaseModel):
    """Performance metrics for monitoring"""
    qps_current: float = Field(..., description="Current queries per second")
    qps_average: float = Field(..., description="Average QPS over time window")
    latency_p50_ms: float = Field(..., description="50th percentile latency")
    latency_p95_ms: float = Field(..., description="95th percentile latency")
    latency_p99_ms: float = Field(..., description="99th percentile latency")
    memory_usage_mb: float = Field(..., description="Current memory usage")
    cache_hit_rate: float = Field(..., description="Cache hit rate percentage")
    mlx_compilation_time_ms: float = Field(..., description="MLX kernel compilation time")


class SystemStatus(BaseModel):
    """System status information"""
    mlx_version: str = Field(..., description="MLX framework version")
    device_info: Dict[str, Any] = Field(..., description="Device information")
    memory_info: Dict[str, float] = Field(..., description="Memory statistics")
    performance_metrics: PerformanceMetrics = Field(..., description="Performance metrics")
    active_stores: int = Field(..., description="Number of active stores")
    total_vectors: int = Field(..., description="Total vectors across all stores")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


# Configuration models for different use cases
class ProductionConfig(VectorStoreConfig):
    """Production-optimized configuration"""
    dimension: int = Field(default=384)
    metric: str = Field(default="cosine")
    index_type: str = Field(default="hnsw")  # Use HNSW for production
    chunk_size: int = Field(default=5000)  # Larger batches
    cache_size: int = Field(default=50000)  # Larger cache
    use_metal: bool = Field(default=True)
    jit_compile: bool = Field(default=True)


class DevelopmentConfig(VectorStoreConfig):
    """Development-friendly configuration"""
    dimension: int = Field(default=384)
    metric: str = Field(default="cosine") 
    index_type: str = Field(default="flat")  # Simpler for debugging
    chunk_size: int = Field(default=1000)
    cache_size: int = Field(default=5000)
    use_metal: bool = Field(default=True)
    jit_compile: bool = Field(default=False)  # Easier debugging


class BenchmarkConfig(VectorStoreConfig):
    """Configuration optimized for benchmarking"""
    dimension: int = Field(default=384)
    metric: str = Field(default="cosine")
    index_type: str = Field(default="flat")
    chunk_size: int = Field(default=10000)  # Large batches
    cache_size: int = Field(default=100000)  # Maximum cache
    use_metal: bool = Field(default=True)
    jit_compile: bool = Field(default=True)


# Utility functions for model validation
def validate_vector_dimension(vectors: List[List[float]], expected_dim: int) -> bool:
    """Validate that all vectors have the expected dimension"""
    return all(len(vec) == expected_dim for vec in vectors)


def validate_metadata_consistency(vectors: List[List[float]], metadata: List[Dict]) -> bool:
    """Validate that vectors and metadata have consistent lengths"""
    return len(vectors) == len(metadata)


def estimate_memory_usage(num_vectors: int, dimension: int) -> float:
    """Estimate memory usage in MB for given vector count and dimension"""
    # float32 = 4 bytes per element
    vector_memory = num_vectors * dimension * 4
    # Estimate metadata overhead (rough approximation)
    metadata_memory = num_vectors * 1024  # 1KB per metadata entry
    
    total_bytes = vector_memory + metadata_memory
    return total_bytes / (1024 * 1024)  # Convert to MB


# Response helpers
def create_error_response(message: str, error_code: Optional[str] = None) -> ErrorResponse:
    """Create standardized error response"""
    import datetime
    return ErrorResponse(
        detail=message,
        error_code=error_code,
        timestamp=datetime.datetime.utcnow().isoformat()
    )


def create_success_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create standardized success response"""
    return {
        "success": True,
        "timestamp": __import__('datetime').datetime.utcnow().isoformat(),
        **data
    }


# Constants for validation
MAX_VECTOR_DIMENSION = 4096
MAX_BATCH_SIZE = 1000
MAX_QUERY_K = 1000
SUPPORTED_METRICS = ["cosine", "euclidean", "dot_product"]
SUPPORTED_INDEX_TYPES = ["flat", "hnsw"]

# Example usage and test data generators
def generate_test_vectors(count: int, dimension: int = 384) -> List[List[float]]:
    """Generate test vectors for development/testing"""
    import random
    return [
        [random.uniform(-1, 1) for _ in range(dimension)]
        for _ in range(count)
    ]


def generate_test_metadata(count: int, categories: List[str] = None) -> List[Dict[str, Any]]:
    """Generate test metadata for development/testing"""
    if categories is None:
        categories = ["tech", "science", "business", "health", "education"]
    
    import random
    return [
        {
            "id": f"test_doc_{i}",
            "category": random.choice(categories),
            "source": "test_data",
            "timestamp": f"2024-01-{(i % 28) + 1:02d}",
            "priority": random.randint(1, 5)
        }
        for i in range(count)
    ]


# MLX-specific model extensions
class MLXOptimizationInfo(BaseModel):
    """Information about MLX optimizations"""
    kernels_compiled: bool = Field(..., description="Whether MLX kernels are compiled")
    metal_acceleration: bool = Field(..., description="Metal GPU acceleration status")
    unified_memory: bool = Field(..., description="Unified memory usage status")
    lazy_evaluation: bool = Field(..., description="Lazy evaluation status")
    compilation_time_ms: float = Field(..., description="Kernel compilation time")
    device_type: str = Field(..., description="MLX device type (cpu/gpu)")
    memory_pool_size_mb: float = Field(..., description="MLX memory pool size")


class MLXPerformanceReport(BaseModel):
    """Detailed MLX performance report"""
    vector_addition_rate: float = Field(..., description="Vectors added per second")
    query_throughput_qps: float = Field(..., description="Queries per second")
    batch_processing_speedup: float = Field(..., description="Batch vs single query speedup")
    memory_efficiency: float = Field(..., description="Memory usage efficiency ratio")
    metal_utilization: float = Field(..., description="Metal GPU utilization percentage")
    compilation_overhead_ms: float = Field(..., description="JIT compilation overhead")
    optimization_info: MLXOptimizationInfo = Field(..., description="Optimization details")


# Export all models for easy importing
__all__ = [
    # Core request/response models
    "VectorAddRequest", "VectorQuery", "BatchQueryRequest", 
    "VectorQueryResponse", "BatchQueryResponse", "VectorAddResponse",
    
    # Configuration models
    "VectorStoreConfig", "CreateStoreRequest", "ProductionConfig", 
    "DevelopmentConfig", "BenchmarkConfig",
    
    # Result models
    "VectorSearchResult", "StoreStatsResponse", "HealthResponse",
    
    # Advanced query models
    "AdvancedQueryRequest", "HybridQueryRequest", "StreamResult",
    
    # Admin models
    "ExportRequest", "ImportRequest", "ExportResponse", "ImportResponse",
    
    # Monitoring models
    "PerformanceMetrics", "SystemStatus", "MLXOptimizationInfo", "MLXPerformanceReport",
    
    # Error models
    "ErrorResponse", "ValidationErrorResponse", "ValidationErrorDetail",
    
    # Benchmark models
    "BenchmarkRequest", "BenchmarkResponse",
    
    # Utility functions
    "validate_vector_dimension", "validate_metadata_consistency", 
    "estimate_memory_usage", "create_error_response", "create_success_response",
    "generate_test_vectors", "generate_test_metadata"
]