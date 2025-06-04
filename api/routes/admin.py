# api/routes/admin.py - Korrekte Reihenfolge:

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any, Union
import asyncio
import json
import time
import zipfile
import io
import os
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import logging
import numpy as np

from service.vector_store import MLXVectorStore, VectorStoreConfig
from service.models import (
    CreateStoreRequest,
    CreateStoreResponse,
    StoreStatsResponse,
    OptimizeResponse,
    ExportRequest,
    ExportResponse,
    ImportRequest,
    ImportResponse,
    BackupRequest,
    BackupResponse,
    RestoreRequest,
    RestoreResponse,
    ErrorResponse
)
from security.auth import verify_api_key, verify_admin_key
from api.routes.vectors import store_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])

# Admin-specific thread pool for I/O operations
admin_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="admin")

# HIER - Nach den Imports definieren:
def create_success_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a standardized success response"""
    return {
        "status": "success",
        "data": data,
        "timestamp": time.time()
    }

def create_error_response(message: str, error_code: str = "ERROR") -> ErrorResponse:
    """Create a standardized error response"""
    return ErrorResponse(
        error=message,
        code=error_code
    )

# Dann folgen die anderen Klassen und Routen...
class StoreInfo(BaseModel):
    """Information about a vector store"""
    user_id: str
    model_id: str
    vector_count: int
    dimension: int
    memory_usage_mb: float
    index_type: str
    metric: str
    created_at: Optional[str] = None
    last_accessed: Optional[str] = None


class BulkStoreOperation(BaseModel):
    """Bulk operation on multiple stores"""
    operation: str  # create, delete, optimize, export
    stores: List[Dict[str, str]]  # [{"user_id": "...", "model_id": "..."}]
    config: Optional[VectorStoreConfig] = None


class StoreMaintenanceRequest(BaseModel):
    """Store maintenance operations"""
    user_id: str
    model_id: str
    operations: List[str]  # ["optimize", "compact", "reindex", "cleanup"]
    force: bool = False


@router.post("/create_store")
async def create_store(
    request: CreateStoreRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """
    Create a new vector store with specified configuration
    """
    try:
        # Use provided config or defaults
        config = request.config or VectorStoreConfig()
        
        # Check if store already exists
        store_key = store_manager.get_store_key(request.user_id, request.model_id)
        if store_key in store_manager._stores:
            raise HTTPException(
                status_code=409, 
                detail=f"Store already exists for {request.user_id}/{request.model_id}"
            )
        
        # Create store asynchronously
        store = await store_manager.get_store(
            request.user_id, 
            request.model_id, 
            config
        )
        
        # Schedule background optimization
        background_tasks.add_task(
            optimize_store_background, 
            request.user_id, 
            request.model_id
        )
        
        logger.info(f"✅ Created store: {request.user_id}/{request.model_id}")
        
        return create_success_response({
            "store_created": True,
            "user_id": request.user_id,
            "model_id": request.model_id,
            "config": config.dict(),
            "store_path": str(store.store_path)
        })
        
    except Exception as e:
        logger.error(f"Failed to create store: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# api/routes/admin.py - Ersetze die delete_store Route (ca. Zeile 82)

@router.delete("/store")
async def delete_store(
    user_id: str,
    model_id: str,
    force: bool = False,
    api_key: str = Depends(verify_admin_key)
):
    """
    Delete a vector store and all its data
    Requires admin privileges
    """
    try:
        store_key = store_manager.get_store_key(user_id, model_id)
        
        if store_key not in store_manager._stores:
            raise HTTPException(
                status_code=404,
                detail=f"Store not found: {user_id}/{model_id}"
            )
        
        store = store_manager._stores[store_key]
        store_path = store.store_path
        vector_count = store.get_stats()['vector_count']
        
        # Safety check for non-empty stores
        if vector_count > 0 and not force:
            raise HTTPException(
                status_code=400,
                detail=f"Store contains {vector_count} vectors. Use force=true to delete."
            )
        
        # Clear store data
        await asyncio.get_event_loop().run_in_executor(
            admin_executor,
            store.clear
        )
        
        # Remove from store manager
        del store_manager._stores[store_key]
        if store_key in store_manager._configs:
            del store_manager._configs[store_key]
        
        # Remove directory
        if store_path.exists():
            await asyncio.get_event_loop().run_in_executor(
                admin_executor,
                lambda: shutil.rmtree(store_path, ignore_errors=True)
            )
        
        logger.info(f"🗑️ Deleted store: {user_id}/{model_id}")
        
        content=create_success_response({
            "store_deleted": True,
            "user_id": user_id,
            "model_id": model_id,
            "vectors_removed": vector_count
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete store: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# api/routes/admin.py - Füge diese Route nach der delete_store Route ein (ca. Zeile 234)

@router.get("/store/stats")
async def get_store_stats(
    user_id: str,
    model_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Get statistics for a specific store"""
    try:
        store = await store_manager.get_store(user_id, model_id)
        stats = store.get_stats()
        
        return {
            "vectors": stats.get('vector_count', 0),
            "metadata": len(store._metadata) if hasattr(store, '_metadata') else 0,
            "dimension": stats.get('dimension', 0),
            "memory_usage_mb": stats.get('memory_usage_mb', 0.0)
        }
        
    except Exception as e:
        logger.error(f"Failed to get store stats: {e}")
        raise HTTPException(status_code=404, detail=f"Store not found: {user_id}/{model_id}")

@router.get("/list_stores")
async def list_stores(
    user_id: Optional[str] = None,
    api_key: str = Depends(verify_api_key)
):
    """
    List all stores or stores for a specific user
    """
    try:
        stores_info = []
        
        for store_key, store in store_manager._stores.items():
            store_user_id, store_model_id = store_key.split("_", 1)
            
            # Filter by user if specified
            if user_id and store_user_id != user_id:
                continue
            
            stats = store.get_stats()
            store_info = StoreInfo(
                user_id=store_user_id,
                model_id=store_model_id,
                vector_count=stats['vector_count'],
                dimension=stats['dimension'],
                memory_usage_mb=stats['memory_usage_mb'],
                index_type=stats['index_type'],
                metric=stats['metric'],
                # Add timestamps if available
                created_at=getattr(store, 'created_at', None),
                last_accessed=getattr(store, 'last_accessed', None)
            )
            stores_info.append(store_info)
        
        return {
            "stores": [store.dict() for store in stores_info],
            "total_stores": len(stores_info),
            "filtered_by_user": user_id is not None
        }
        
    except Exception as e:
        logger.error(f"Failed to list stores: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export_zip", response_model=ExportResponse)
async def export_store_zip(
    request: ExportRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Export store data as ZIP file
    Includes vectors, metadata, and configuration
    """
    start_time = time.time()
    
    try:
        # Get store
        store = await store_manager.get_store(request.user_id, request.model_id)
        
        if store.get_stats()['vector_count'] == 0:
            raise HTTPException(
                status_code=400,
                detail="Cannot export empty store"
            )
        
        # Create export in thread pool
        export_path = await asyncio.get_event_loop().run_in_executor(
            admin_executor,
            _create_store_export,
            store,
            request
        )
        
        # Get file size
        file_size_mb = os.path.getsize(export_path) / (1024 * 1024)
        export_time = (time.time() - start_time) * 1000
        
        logger.info(f"📦 Exported store: {request.user_id}/{request.model_id}")
        
        return ExportResponse(
            success=True,
            export_path=str(export_path),
            file_size_mb=file_size_mb,
            vectors_exported=store.get_stats()['vector_count'],
            export_time_ms=export_time
        )
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download_export/{user_id}/{model_id}")
async def download_export(
    user_id: str,
    model_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Download exported store ZIP file
    """
    try:
        export_path = Path(f"./exports/{user_id}_{model_id}_export.zip")
        
        if not export_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Export file not found. Please export first."
            )
        
        return FileResponse(
            path=export_path,
            filename=f"{user_id}_{model_id}_vectorstore.zip",
            media_type="application/zip"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/import_zip", response_model=ImportResponse)
async def import_store_zip(
    request: ImportRequest,
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    """
    Import store data from ZIP file
    Can overwrite existing stores if specified
    """
    start_time = time.time()
    
    try:
        # Validate file
        if not file.filename.endswith('.zip'):
            raise HTTPException(
                status_code=400,
                detail="File must be a ZIP archive"
            )
        
        # Check if store exists
        store_key = store_manager.get_store_key(request.user_id, request.model_id)
        store_exists = store_key in store_manager._stores
        
        if store_exists and not request.overwrite_existing:
            raise HTTPException(
                status_code=409,
                detail="Store exists. Set overwrite_existing=true to replace."
            )
        
        # Read ZIP file
        zip_content = await file.read()
        
        # Import in thread pool
        import_result = await asyncio.get_event_loop().run_in_executor(
            admin_executor,
            _import_store_from_zip,
            zip_content,
            request,
            store_exists
        )
        
        import_time = (time.time() - start_time) * 1000
        
        logger.info(f"📥 Imported store: {request.user_id}/{request.model_id}")
        
        return ImportResponse(
            success=True,
            vectors_imported=import_result['vectors_imported'],
            metadata_imported=import_result['metadata_imported'],
            import_time_ms=import_time,
            store_recreated=store_exists
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Import failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize_store")
async def optimize_store(
    user_id: str,
    model_id: str,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """
    Optimize store for better performance
    Includes index optimization and memory cleanup
    """
    try:
        store = await store_manager.get_store(user_id, model_id)
        
        if store.get_stats()['vector_count'] == 0:
            raise HTTPException(
                status_code=400,
                detail="Cannot optimize empty store"
            )
        
        # Schedule optimization in background
        background_tasks.add_task(
            optimize_store_background,
            user_id,
            model_id
        )
        
        return create_success_response({
            "optimization_scheduled": True,
            "user_id": user_id,
            "model_id": model_id,
            "message": "Store optimization running in background"
        })
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bulk_operation")
async def bulk_store_operation(
    request: BulkStoreOperation,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_admin_key)
):
    """
    Perform bulk operations on multiple stores
    Requires admin privileges
    """
    try:
        results = []
        
        for store_info in request.stores:
            user_id = store_info['user_id']
            model_id = store_info['model_id']
            
            try:
                if request.operation == "create":
                    config = request.config or VectorStoreConfig()
                    await store_manager.get_store(user_id, model_id, config)
                    results.append({"user_id": user_id, "model_id": model_id, "status": "created"})
                
                elif request.operation == "optimize":
                    background_tasks.add_task(optimize_store_background, user_id, model_id)
                    results.append({"user_id": user_id, "model_id": model_id, "status": "optimization_scheduled"})
                
                elif request.operation == "delete":
                    store_key = store_manager.get_store_key(user_id, model_id)
                    if store_key in store_manager._stores:
                        store = store_manager._stores[store_key]
                        store.clear()
                        del store_manager._stores[store_key]
                    results.append({"user_id": user_id, "model_id": model_id, "status": "deleted"})
                
                else:
                    results.append({"user_id": user_id, "model_id": model_id, "status": "unsupported_operation"})
                    
            except Exception as e:
                results.append({"user_id": user_id, "model_id": model_id, "status": "error", "error": str(e)})
        
        return create_success_response({
            "bulk_operation": request.operation,
            "total_stores": len(request.stores),
            "results": results
        })
        
    except Exception as e:
        logger.error(f"Bulk operation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/maintenance")
async def store_maintenance(
    request: StoreMaintenanceRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_admin_key)
):
    """
    Perform maintenance operations on a store
    """
    try:
        store = await store_manager.get_store(request.user_id, request.model_id)
        
        maintenance_results = []
        
        for operation in request.operations:
            if operation == "optimize":
                background_tasks.add_task(optimize_store_background, request.user_id, request.model_id)
                maintenance_results.append({"operation": "optimize", "status": "scheduled"})
                
            elif operation == "compact":
                # Force garbage collection and memory cleanup
                background_tasks.add_task(_compact_store, store)
                maintenance_results.append({"operation": "compact", "status": "scheduled"})
                
            elif operation == "reindex":
                # Rebuild search indices
                background_tasks.add_task(_reindex_store, store)
                maintenance_results.append({"operation": "reindex", "status": "scheduled"})
                
            elif operation == "cleanup":
                # Clean temporary files and caches
                background_tasks.add_task(_cleanup_store, store)
                maintenance_results.append({"operation": "cleanup", "status": "scheduled"})
                
            else:
                maintenance_results.append({"operation": operation, "status": "unsupported"})
        
        return create_success_response({
            "maintenance_scheduled": True,
            "user_id": request.user_id,
            "model_id": request.model_id,
            "operations": maintenance_results
        })
        
    except Exception as e:
        logger.error(f"Maintenance failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/store_analytics/{user_id}/{model_id}")
async def get_store_analytics(
    user_id: str,
    model_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Get detailed analytics for a specific store
    """
    try:
        store = await store_manager.get_store(user_id, model_id)
        stats = store.get_stats()
        
        # Calculate additional analytics
        analytics = {
            "basic_stats": stats,
            "performance_metrics": {
                "estimated_qps": min(1000, stats['vector_count'] / 10),  # Rough estimate
                "memory_efficiency": stats['memory_usage_mb'] / max(stats['vector_count'], 1),
                "index_efficiency": "optimal" if stats['index_type'] == "hnsw" else "basic"
            },
            "recommendations": []
        }
        
        # Add recommendations based on stats
        if stats['vector_count'] > 10000 and stats['index_type'] == "flat":
            analytics["recommendations"].append("Consider switching to HNSW index for better performance")
        
        if stats['memory_usage_mb'] > 1000:
            analytics["recommendations"].append("Large memory usage detected, consider optimization")
        
        return analytics
        
    except Exception as e:
        logger.error(f"Analytics failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Background task functions
async def optimize_store_background(user_id: str, model_id: str):
    """Background store optimization"""
    try:
        store = await store_manager.get_store(user_id, model_id)
        await asyncio.get_event_loop().run_in_executor(
            admin_executor,
            store.optimize
        )
        logger.info(f"✅ Optimized store: {user_id}/{model_id}")
    except Exception as e:
        logger.error(f"Background optimization failed: {e}")


async def _compact_store(store: MLXVectorStore):
    """Compact store data"""
    try:
        # Force memory cleanup and reorganization
        store._save_store()
        logger.info("✅ Store compaction completed")
    except Exception as e:
        logger.error(f"Store compaction failed: {e}")


async def _reindex_store(store: MLXVectorStore):
    """Rebuild store indices"""
    try:
        # Future: HNSW index rebuilding
        store.optimize()
        logger.info("✅ Store reindexing completed")
    except Exception as e:
        logger.error(f"Store reindexing failed: {e}")


async def _cleanup_store(store: MLXVectorStore):
    """Clean up temporary files"""
    try:
        # Clean temporary files and caches
        temp_files = store.store_path.glob("*.tmp")
        for temp_file in temp_files:
            temp_file.unlink(missing_ok=True)
        logger.info("✅ Store cleanup completed")
    except Exception as e:
        logger.error(f"Store cleanup failed: {e}")


# Helper functions for import/export
def _create_store_export(store: MLXVectorStore, request: ExportRequest) -> Path:
    """Create ZIP export of store data"""
    export_dir = Path("./exports")
    export_dir.mkdir(exist_ok=True)
    
    export_path = export_dir / f"{request.user_id}_{request.model_id}_export.zip"
    
    with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED, 
                        compresslevel=request.compression_level) as zf:
        
        # Add vector data
        if (store.store_path / "vectors.npz").exists():
            zf.write(store.store_path / "vectors.npz", "vectors.npz")
        
        # Add metadata if requested
        if request.include_metadata and (store.store_path / "metadata.jsonl").exists():
            zf.write(store.store_path / "metadata.jsonl", "metadata.jsonl")
        
        # Add configuration
        if (store.store_path / "config.json").exists():
            zf.write(store.store_path / "config.json", "config.json")
        
        # Add export manifest
        manifest = {
            "export_version": "1.0",
            "user_id": request.user_id,
            "model_id": request.model_id,
            "export_timestamp": time.time(),
            "vector_count": store.get_stats()['vector_count'],
            "dimension": store.get_stats()['dimension']
        }
        
        manifest_json = json.dumps(manifest, indent=2)
        zf.writestr("manifest.json", manifest_json)
    
    return export_path


def _import_store_from_zip(zip_content: bytes, request: ImportRequest, store_exists: bool) -> Dict[str, int]:
    """Import store data from ZIP content"""
    
    with zipfile.ZipFile(io.BytesIO(zip_content), 'r') as zf:
        # Read manifest
        manifest = json.loads(zf.read("manifest.json").decode())
        
        # Create store directory
        store_path = Path(f"~/.team_mind_data/vector_stores/{request.user_id}/{request.model_id}")
        store_path.mkdir(parents=True, exist_ok=True)
        
        # Extract files
        vectors_imported = 0
        metadata_imported = 0
        
        if "vectors.npz" in zf.namelist():
            zf.extract("vectors.npz", store_path)
            vectors_imported = manifest.get('vector_count', 0)
        
        if "metadata.jsonl" in zf.namelist():
            zf.extract("metadata.jsonl", store_path)
            # Count metadata entries
            metadata_content = zf.read("metadata.jsonl").decode()
            metadata_imported = len([line for line in metadata_content.split('\n') if line.strip()])
        
        if "config.json" in zf.namelist():
            zf.extract("config.json", store_path)
    
    return {
        "vectors_imported": vectors_imported,
        "metadata_imported": metadata_imported
    }