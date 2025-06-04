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
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "vectors": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            "metadata": [
                {"id": "doc1", "text": "Hello world"},
                {"id": "doc2", "text": "Another document"}
            ]
        }
    })
    
    vectors: List[List[float]] = Field(
        ...,
        description="List of vectors to add"
    )
    metadata: List[Dict[str, Any]] = Field(
        ...,
        description="Metadata for each vector"
    )
    
    @field_validator('vectors', 'metadata')
    @classmethod
    def validate_same_length(cls, v, info):
        if 'vectors' in info.data and 'metadata' in info.data:
            if len(info.data['vectors']) != len(info.data['metadata']):
                raise ValueError('Vectors and metadata must have the same length')
        return v


class VectorQuery(BaseModel):
    """Request model for vector similarity search"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "vector": [0.1, 0.2, 0.3],
            "k": 10,
            "filter": {"category": "science"}
        }
    })
    
    vector: List[float] = Field(
        ...,
        description="Query vector"
    )
    k: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Number of results to return"
    )
    filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata filter"
    )


class BatchQueryRequest(BaseModel):
    """Request model for batch queries"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "queries": [
                {"vector": [0.1, 0.2, 0.3], "k": 5},
                {"vector": [0.4, 0.5, 0.6], "k": 10}
            ]
        }
    })
    
    queries: List[VectorQuery] = Field(
        ...,
        description="List of queries to process"
    )


class SearchResult(BaseModel):
    """Single search result"""
    metadata: Dict[str, Any] = Field(
        ...,
        description="Metadata of the matched vector"
    )
    score: float = Field(
        ...,
        description="Similarity score"
    )
    index: Optional[int] = Field(
        default=None,
        description="Index in the vector store"
    )


class QueryResponse(BaseModel):
    """Response model for vector queries"""
    results: List[SearchResult] = Field(
        ...,
        description="Search results"
    )
    query_time_ms: float = Field(
        ...,
        description="Query execution time in milliseconds"
    )


class BatchQueryResponse(BaseModel):
    """Response model for batch queries"""
    results: List[QueryResponse] = Field(
        ...,
        description="Results for each query"
    )
    total_time_ms: float = Field(
        ...,
        description="Total batch processing time"
    )


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
    name: str = Field(
        ...,
        pattern="^[a-zA-Z0-9_-]+$",
        description="Store name (alphanumeric, underscores, hyphens)"
    )
    dimension: int = Field(
        ...,
        ge=1,
        le=4096,
        description="Vector dimension"
    )
    metric: MetricType = Field(
        default=MetricType.COSINE,
        description="Similarity metric"
    )
    index_type: IndexType = Field(
        default=IndexType.FLAT,
        description="Index type"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional store metadata"
    )


class DeleteVectorsRequest(BaseModel):
    """Request model for deleting vectors"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "indices": [0, 1, 2],
            "filter": {"category": "outdated"}
        }
    })
    
    indices: Optional[List[int]] = Field(
        default=None,
        description="Indices of vectors to delete"
    )
    filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Delete vectors matching this metadata filter"
    )


class DeleteVectorsResponse(BaseModel):
    """Response model for delete operations"""
    deleted_count: int = Field(
        ...,
        description="Number of vectors deleted"
    )
    remaining_count: int = Field(
        ...,
        description="Number of vectors remaining in store"
    )


class OptimizeRequest(BaseModel):
    """Request model for optimization operations"""
    rebuild_index: bool = Field(
        default=True,
        description="Whether to rebuild the index"
    )
    compact_storage: bool = Field(
        default=True,
        description="Whether to compact storage"
    )


class CreateStoreRequest(BaseModel):
    """Request model for creating a new vector store"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "name": "my_store",
            "dimension": 384,
            "metric": "cosine"
        }
    })
    
    name: str = Field(
        ...,
        pattern="^[a-zA-Z0-9_-]+$",
        description="Store name (alphanumeric, underscores, hyphens)"
    )
    dimension: int = Field(
        ...,
        ge=1,
        le=4096,
        description="Vector dimension"
    )
    metric: MetricType = Field(
        default=MetricType.COSINE,
        description="Similarity metric"
    )


class CreateStoreResponse(BaseModel):
    """Response model for store creation"""
    name: str
    message: str
    config: VectorStoreConfig


class ExportRequest(BaseModel):
    """Request model for exporting store data"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "format": "npz",
            "include_metadata": True
        }
    })
    
    format: str = Field(
        default="npz",
        pattern="^(npz|json|parquet)$",
        description="Export format"
    )
    include_metadata: bool = Field(
        default=True,
        description="Whether to include metadata"
    )


class ImportRequest(BaseModel):
    """Request model for importing store data"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "source_path": "/path/to/data.npz",
            "format": "npz",
            "merge": False
        }
    })
    
    source_path: str = Field(
        ...,
        description="Path to import data from"
    )
    format: str = Field(
        default="npz",
        pattern="^(npz|json|parquet)$",
        description="Import format"
    )
    merge: bool = Field(
        default=False,
        description="Whether to merge with existing data"
    )


class ExportResponse(BaseModel):
    """Response model for export operations"""
    message: str
    export_path: str
    format: str
    vectors_exported: int
    file_size_mb: float


class ImportResponse(BaseModel):
    """Response model for import operations"""
    message: str
    vectors_imported: int
    format: str
    merge_mode: bool


class BackupResponse(BaseModel):
    """Response model for backup operations"""
    message: str
    backup_path: str
    backup_size_mb: float
    timestamp: str


class RestoreResponse(BaseModel):
    """Response model for restore operations"""
    message: str
    vectors_restored: int
    restore_time_ms: float


class BackupRequest(BaseModel):
    """Request model for backup operations"""
    destination_path: Optional[str] = Field(
        default=None,
        description="Backup destination path"
    )
    compress: bool = Field(
        default=True,
        description="Whether to compress backup"
    )


class RestoreRequest(BaseModel):
    """Request model for restore operations"""
    backup_path: str = Field(
        ...,
        description="Path to backup file"
    )
    overwrite: bool = Field(
        default=False,
        description="Whether to overwrite existing data"
    )


class StoreStatsResponse(BaseModel):
    """Response model for store statistics"""
    name: str
    vector_count: int
    dimension: int
    metric: str
    memory_usage_mb: float
    index_type: str
    mlx_device: str
    unified_memory: bool


class OptimizeResponse(BaseModel):
    """Response model for optimization operations"""
    message: str
    optimization_time_ms: float
    vectors_optimized: int


class StoreListResponse(BaseModel):
    """Response model for listing stores"""
    stores: List[VectorStoreInfo] = Field(
        ...,
        description="List of available stores"
    )
    total_memory_mb: float = Field(
        ...,
        description="Total memory usage across all stores"
    )


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(
        default="healthy",
        description="Service health status"
    )
    mlx_version: str = Field(
        ...,
        description="MLX version"
    )
    metal_available: bool = Field(
        ...,
        description="Whether Metal acceleration is available"
    )
    stores_count: int = Field(
        ...,
        description="Number of active stores"
    )
    uptime_seconds: float = Field(
        ...,
        description="Service uptime in seconds"
    )


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(
        ...,
        description="Error message"
    )
    detail: Optional[str] = Field(
        default=None,
        description="Detailed error information"
    )
    code: Optional[str] = Field(
        default=None,
        description="Error code"
    )


# Validation utilities
def validate_vector_dimension(vector: List[float], expected_dim: int) -> bool:
    """Validate vector dimension"""
    return len(vector) == expected_dim


def validate_vectors_batch(vectors: List[List[float]]) -> bool:
    """Validate a batch of vectors have consistent dimensions"""
    if not vectors:
        return True
    
    dim = len(vectors[0])
    return all(len(v) == dim for v in vectors)


# Export all models
__all__ = [
    'MetricType',
    'IndexType',
    'VectorAddRequest',
    'VectorQuery',
    'BatchQueryRequest',
    'SearchResult',
    'QueryResponse',
    'BatchQueryResponse',
    'VectorStoreInfo',
    'VectorStoreConfig',
    'CreateStoreRequest',
    'CreateStoreResponse',
    'StoreStatsResponse',
    'OptimizeResponse',
    'DeleteVectorsRequest',
    'DeleteVectorsResponse',
    'OptimizeRequest',
    'StoreListResponse',
    'HealthResponse',
    'ErrorResponse',
    'ExportRequest',
    'ExportResponse',
    'ImportRequest',
    'ImportResponse',
    'BackupRequest',
    'BackupResponse',
    'RestoreRequest',
    'RestoreResponse',
    'validate_vector_dimension',
    'validate_vectors_batch'
]