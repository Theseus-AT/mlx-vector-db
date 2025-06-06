"""
Pydantic models for MLX Vector DB API
Korrigierte Version für Pydantic v2 Kompatibilität
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from enum import Enum
import numpy as np

class MetricType(str, Enum):
    """Supported similarity metrics"""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"

class IndexType(str, Enum):
    """Supported index types"""
    FLAT = "flat"
    HNSW = "hnsw"

class VectorAddRequest(BaseModel):
    """Request model for adding vectors"""
    user_id: str
    model_id: str
    vectors: List[List[float]] = Field(..., description="List of vectors to add")
    metadata: List[Dict[str, Any]] = Field(..., description="Metadata for each vector")
    
    @model_validator(mode='after')
    def validate_same_length(self):
        """Validate that vectors and metadata have the same length"""
        if len(self.vectors) != len(self.metadata):
            raise ValueError('Vectors and metadata must have the same length')
        return self

class VectorQuery(BaseModel):
    """Request model for vector similarity search"""
    user_id: str
    model_id: str
    query: List[float] = Field(..., description="Query vector")
    k: int = Field(default=10, ge=1, le=1000)
    filter_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filter")

class BatchQueryRequest(BaseModel):
    """Request model for batch queries"""
    user_id: str
    model_id: str
    queries: List[List[float]]
    k: int = 10

class SearchResult(BaseModel):
    """Single search result"""
    metadata: Dict[str, Any]
    score: float
    index: Optional[int] = None

class QueryResponse(BaseModel):
    """Response model for vector queries"""
    results: List[SearchResult]
    query_time_ms: float

class BatchQueryResponse(BaseModel):
    """Response model for batch queries"""
    results: List[QueryResponse]
    total_time_ms: float

class VectorStoreInfo(BaseModel):
    """Information about a vector store"""
    name: str
    dimension: int
    metric: MetricType
    vector_count: int
    memory_usage_mb: float
    index_type: IndexType
    metadata: Dict[str, Any] = Field(default_factory=dict)

class VectorStoreConfig(BaseModel):
    """Configuration for creating a vector store"""
    name: str = Field(..., pattern="^[a-zA-Z0-9_-]+$")
    dimension: int = Field(..., ge=1, le=4096)
    metric: MetricType = Field(default=MetricType.COSINE)
    index_type: IndexType = Field(default=IndexType.FLAT)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DeleteVectorsRequest(BaseModel):
    """Request model for deleting vectors"""
    indices: Optional[List[int]] = None
    filter: Optional[Dict[str, Any]] = None

class DeleteVectorsResponse(BaseModel):
    """Response model for delete operations"""
    deleted_count: int
    remaining_count: int

class OptimizeRequest(BaseModel):
    """Request model for optimization operations"""
    rebuild_index: bool = True
    compact_storage: bool = True

class CreateStoreRequest(BaseModel):
    """Request model for creating a new vector store"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "user_id": "my_user",
            "model_id": "my_model",
            "dimension": 384,
            "metric": "cosine"
        }
    })
    
    user_id: str = Field(..., description="User ID for the store")
    model_id: str = Field(..., description="Model ID for the store")
    dimension: int = Field(default=384, ge=1, le=4096, description="Vector dimension")
    metric: MetricType = Field(default=MetricType.COSINE, description="Similarity metric")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Advanced store configuration")

class CreateStoreResponse(BaseModel):
    """Response model for store creation"""
    status: str
    data: Dict[str, Any]

class StoreStatsResponse(BaseModel):
    """Response model for store statistics"""
    user_id: str
    model_id: str
    vector_count: int
    dimension: int
    metric: str
    memory_usage_mb: float
    index_type: str

class OptimizeResponse(BaseModel):
    """Response model for optimization operations"""
    message: str
    optimization_time_ms: float
    vectors_optimized: int

class HealthResponse(BaseModel):
    """Response model for health checks"""
    status: str = "healthy"
    mlx_version: str = "0.25.2"
    metal_available: bool = True
    stores_count: int = 0
    uptime_seconds: float = 0.0

class BenchmarkRequest(BaseModel):
    """Request model for performance benchmarks"""
    user_id: str
    model_id: str
    num_vectors: int = Field(default=1000, ge=1, le=1000000)
    num_queries: int = Field(default=100, ge=1, le=10000)

class BenchmarkResponse(BaseModel):
    """Response model for benchmarks"""
    benchmark_results: Dict[str, Any]
    mlx_optimized: bool = True
    performance_target: str = "1000+ QPS"

class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None
    timestamp: Optional[float] = None

# Performance monitoring models
class PerformanceMetrics(BaseModel):
    """Performance metrics model"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time_ms: float = 0.0
    current_qps: float = 0.0
    uptime_seconds: float = 0.0

class SystemInfo(BaseModel):
    """System information model"""
    platform: str
    processor: str
    python_version: str
    mlx_version: str = "0.25.2"
    mlx_device: str
    memory_total_gb: float
    memory_available_gb: float

# Configuration models
class ServerConfig(BaseModel):
    """Server configuration model"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    log_level: str = "info"
    enable_docs: bool = True
    enable_metrics: bool = True

class MLXConfig(BaseModel):
    """MLX-specific configuration"""
    enable_metal: bool = True
    enable_compilation: bool = True
    memory_fraction: float = 0.8
    warmup_on_startup: bool = True

# Export all models
__all__ = [
    "MetricType",
    "IndexType", 
    "VectorAddRequest",
    "VectorQuery",
    "BatchQueryRequest",
    "SearchResult",
    "QueryResponse",
    "BatchQueryResponse",
    "VectorStoreInfo",
    "VectorStoreConfig",
    "DeleteVectorsRequest",
    "DeleteVectorsResponse",
    "OptimizeRequest",
    "CreateStoreRequest",
    "CreateStoreResponse",
    "StoreStatsResponse",
    "OptimizeResponse",
    "HealthResponse",
    "BenchmarkRequest",
    "BenchmarkResponse",
    "ErrorResponse",
    "PerformanceMetrics",
    "SystemInfo",
    "ServerConfig",
    "MLXConfig"
]