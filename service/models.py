"""
Pydantic models for MLX Vector DB API
Updated for Pydantic v2 compatibility
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator
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
    
    @field_validator('vectors', 'metadata')
    @classmethod
    def validate_same_length(cls, v, info):
        if 'vectors' in info.data and 'metadata' in info.data:
            if len(info.data['vectors']) != len(info.data['metadata']):
                raise ValueError('Vectors and metadata must have the same length')
        return v


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


# KORRIGIERT: Dieses Modell wurde an die tatsächliche Verwendung angepasst.
# Es verwendet jetzt user_id und model_id anstelle von 'name'.
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
    dimension: int = Field(..., ge=1, le=4096, description="Vector dimension")
    metric: MetricType = Field(default=MetricType.COSINE, description="Similarity metric")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Advanced store configuration")


class CreateStoreResponse(BaseModel):
    """Response model for store creation"""
    status: str
    data: Dict[str, Any]


class ExportRequest(BaseModel):
    """Request model for exporting store data"""
    user_id: str
    model_id: str
    format: str = Field(default="npz", pattern="^(npz|json|parquet)$")
    include_metadata: bool = True
    compression_level: int = Field(default=zipfile.ZIP_DEFLATED, ge=0, le=9)


class ImportRequest(BaseModel):
    """Request model for importing store data"""
    user_id: str
    model_id: str
    format: str = Field(default="npz", pattern="^(npz|json|parquet)$")
    overwrite_existing: bool = False


class ExportResponse(BaseModel):
    success: bool
    export_path: str
    file_size_mb: float
    vectors_exported: int
    export_time_ms: float

class ImportResponse(BaseModel):
    success: bool
    vectors_imported: int
    metadata_imported: int
    import_time_ms: float
    store_recreated: bool


class BackupResponse(BaseModel):
    message: str
    backup_path: str
    backup_size_mb: float
    timestamp: str


class RestoreResponse(BaseModel):
    message: str
    vectors_restored: int
    restore_time_ms: float


class BackupRequest(BaseModel):
    destination_path: Optional[str] = None
    compress: bool = True


class RestoreRequest(BaseModel):
    backup_path: str
    overwrite: bool = False


class StoreStatsResponse(BaseModel):
    name: str
    vector_count: int
    dimension: int
    metric: str
    memory_usage_mb: float
    index_type: str
    mlx_device: str
    unified_memory: bool


class OptimizeResponse(BaseModel):
    message: str
    optimization_time_ms: float
    vectors_optimized: int


class StoreListResponse(BaseModel):
    stores: List[VectorStoreInfo]
    total_memory_mb: float


class HealthResponse(BaseModel):
    status: str = "healthy"
    mlx_version: str
    metal_available: bool
    stores_count: int
    uptime_seconds: float


class BenchmarkRequest(BaseModel):
    user_id: str
    model_id: str
    num_vectors: int = Field(default=1000, ge=1, le=1000000)
    num_queries: int = Field(default=100, ge=1, le=10000)


class BenchmarkResponse(BaseModel):
    benchmark_results: Dict[str, Any]
    mlx_optimized: bool = True
    performance_target: str = "1000+ QPS"


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None