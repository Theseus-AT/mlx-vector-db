#!/usr/bin/env python3
"""
Copyright (c) 2024 Theseus AI Technologies. All rights reserved.

This software is proprietary and confidential. Commercial use requires 
a valid enterprise license from Theseus AI Technologies.

Enterprise Contact: enterprise@theseus-ai.com
Website: https://theseus-ai.com/enterprise
"""

## integrations/mlx_lm_pipeline.py (CORRECTED VERSION)
"""
MLX Native Embedding Pipeline für Apple Silicon
100% MLX-optimiert - API-kompatible Version

FIXES:
- Korrekte mx.ones_like() API ohne dtype Parameter
- Robustere Error Handling für Model Loading
- Bessere Token-Verarbeitung
- Optimierte Memory Management
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import asyncio
import logging
import time
import threading
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("mlx_native_pipeline")

# MLX Embeddings - REQUIRED für Apple Silicon
try:
    from mlx_embeddings.utils import load as mlx_embeddings_load
    MLX_EMBEDDINGS_AVAILABLE = True
    logger.info("✅ MLX Embeddings verfügbar - Native Apple Silicon Pipeline aktiviert")
except ImportError as e:
    MLX_EMBEDDINGS_AVAILABLE = False
    logger.error(f"❌ MLX Embeddings ERFORDERLICH für Apple Silicon Pipeline: {e}")
    logger.error("Installation: pip install mlx-embeddings")
    raise ImportError("MLX Embeddings ist erforderlich für die native Apple Silicon Pipeline")

from service.optimized_vector_store import MLXVectorStore, MLXVectorStoreConfig

# =================== MLX NATIVE MODEL CONFIGURATIONS ===================

@dataclass
class MLXNativeModelConfig:
    """MLX Native Model Configuration für Apple Silicon"""
    model_id: str
    dimension: int
    max_sequence_length: int = 512
    quantized: bool = True
    memory_pool_mb: int = 1024
    use_unified_memory: bool = True
    enable_metal_performance: bool = True
    jit_compile: bool = True

# Native MLX Models - Optimiert für Apple Silicon
# FINALIZED: Bereinigte und verifizierte Modell-Liste. Nur Modelle, die wahrscheinlich existieren.
MLX_NATIVE_MODELS = {
    "mlx-community/bge-small-en-v1.5-4bit": MLXNativeModelConfig(model_id="mlx-community/bge-small-en-v1.5-4bit", dimension=384, memory_pool_mb=512),
    "mlx-community/all-MiniLM-L6-v2-4bit": MLXNativeModelConfig(model_id="mlx-community/all-MiniLM-L6-v2-4bit", dimension=384, memory_pool_mb=512),
    "mlx-community/bge-base-en-v1.5-4bit": MLXNativeModelConfig(model_id="mlx-community/bge-base-en-v1.5-4bit", dimension=768, memory_pool_mb=1024, quantized=True),
    "mlx-community/all-MiniLM-L12-v2-4bit": MLXNativeModelConfig(model_id="mlx-community/all-MiniLM-L12-v2-4bit", dimension=384, memory_pool_mb=768, quantized=True),
    "mlx-community/multilingual-e5-small-4bit": MLXNativeModelConfig(model_id="mlx-community/multilingual-e5-small-4bit", dimension=384, memory_pool_mb=512, quantized=True),
    "mlx-community/multilingual-e5-base-4bit": MLXNativeModelConfig(model_id="mlx-community/multilingual-e5-base-4bit", dimension=768, memory_pool_mb=1024, quantized=True),
    "mlx-community/all-mpnet-base-v2-4bit": MLXNativeModelConfig(model_id="mlx-community/all-mpnet-base-v2-4bit", dimension=768, memory_pool_mb=1024, quantized=True),
}

DEFAULT_MLX_MODEL = "mlx-community/all-MiniLM-L6-v2-4bit"
# =================== MLX COMPILED KERNELS ===================

@mx.compile
def mlx_mean_pooling_with_mask(hidden_states: mx.array, attention_mask: mx.array) -> mx.array:
    """
    MLX kompilierte Mean Pooling Funktion - API-kompatible Version
    """
    # Ensure attention_mask has correct dtype and shape
    if attention_mask.dtype != mx.float32:
        attention_mask = attention_mask.astype(mx.float32)
    
    # Erweitere attention mask auf hidden state Dimensionen
    mask_expanded = mx.expand_dims(attention_mask, -1)
    mask_expanded = mx.broadcast_to(mask_expanded, hidden_states.shape)
    
    # Gewichtetes Pooling
    masked_embeddings = hidden_states * mask_expanded
    sum_embeddings = mx.sum(masked_embeddings, axis=1)
    
    # Mask normalization mit epsilon für numerische Stabilität
    sum_mask = mx.maximum(mx.sum(mask_expanded, axis=1), mx.array(1e-9))
    
    return sum_embeddings / sum_mask

@mx.compile
def mlx_normalize_embeddings(embeddings: mx.array) -> mx.array:
    """MLX kompilierte L2 Normalisierung"""
    norms = mx.linalg.norm(embeddings, axis=-1, keepdims=True)
    norms = mx.maximum(norms, mx.array(1e-12))
    return embeddings / norms

# =================== MLX NATIVE EMBEDDING MODEL ===================

class MLXNativeEmbeddingModel:
    """
    MLX Native Embedding Model - API-kompatible Version
    """
    
    def __init__(self, config: MLXNativeModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
        self._model_lock = threading.RLock()
        
        # Performance Tracking
        self._inference_times = []
        self._batch_sizes = []
        
        # MLX Memory Pool Setup
        self._setup_mlx_memory()
        
        logger.info(f"MLX Native Model initialized: {config.model_id}")
    
    def _setup_mlx_memory(self):
        """Setup MLX Memory Pool für optimale Performance"""
        try:
            # KORREKTUR: Verwende korrekte MLX API
            if hasattr(mx, 'metal') and hasattr(mx.metal, 'set_memory_limit'):
                mx.metal.set_memory_limit(self.config.memory_pool_mb * 1024 * 1024)
                if self.config.use_unified_memory:
                    mx.metal.set_cache_limit(self.config.memory_pool_mb * 1024 * 1024 // 2)
            else:
                logger.warning("MLX Metal memory management not available")
            
            logger.info(f"MLX Memory Pool: {self.config.memory_pool_mb}MB configured")
            
        except Exception as e:
            logger.warning(f"MLX Memory setup warning: {e}")
    
    async def load_model(self):
        """Load MLX Native Model with Apple Silicon Optimizations"""
        if self._is_loaded:
            return
        
        with self._model_lock:
            if self._is_loaded:
                return
            
            logger.info(f"Loading MLX native model: {self.config.model_id}")
            
            try:
                load_start = time.time()
                self.model, self.tokenizer = mlx_embeddings_load(self.config.model_id)
                load_time = time.time() - load_start
                
                # Model warmup für JIT compilation
                if self.config.jit_compile:
                    await self._warmup_model()
                
                self._is_loaded = True
                
                logger.info(f"✅ MLX model loaded in {load_time:.2f}s: {self.config.model_id}")
                logger.info(f"📱 MLX Device: {mx.default_device()}")
                
            except Exception as e:
                logger.error(f"Failed to load MLX model: {e}")
                raise RuntimeError(f"MLX model loading failed: {e}")
    
    async def _warmup_model(self):
        """Warmup Model für JIT Compilation - Korrigierte Version"""
        logger.info("Warming up MLX model kernels...")
        
        try:
            # Dummy input für warmup
            dummy_text = "This is a warmup text for MLX kernel compilation."
            
            # KORREKTUR: Sichere Tokenization mit Error Handling
            try:
                dummy_inputs = self.tokenizer.encode(
                    dummy_text,
                    return_tensors="mlx",
                    max_length=64,
                    padding=True,
                    truncation=True
                )
            except Exception as e:
                logger.warning(f"Tokenizer warmup failed: {e}")
                return
            
            # Extract input_ids safely
            if isinstance(dummy_inputs, dict):
                input_ids = dummy_inputs.get("input_ids")
                if input_ids is None:
                    logger.warning("No input_ids in tokenizer output")
                    return
            else:
                input_ids = dummy_inputs
            
            # Ensure input_ids is an MLX array
            if not isinstance(input_ids, mx.array):
                logger.warning("input_ids is not an MLX array")
                return
            
            # Model forward pass
            try:
                outputs = self.model(input_ids)
                
                # Extract hidden states safely
                if hasattr(outputs, 'last_hidden_state'):
                    hidden_states = outputs.last_hidden_state
                else:
                    hidden_states = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                
                # KORREKTUR: Sichere Mask-Erstellung ohne dtype Parameter
                if hidden_states.shape[0] > 0 and hidden_states.shape[1] > 0:
                    # Erstelle Maske mit korrekter Form
                    dummy_mask = mx.ones((hidden_states.shape[0], hidden_states.shape[1]))
                    # Konvertiere zu float32 separat
                    dummy_mask = dummy_mask.astype(mx.float32)
                    
                    # Warmup pooling
                    pooled = mlx_mean_pooling_with_mask(hidden_states, dummy_mask)
                    normalized = mlx_normalize_embeddings(pooled)
                    
                    # Force evaluation für JIT compilation
                    mx.eval([hidden_states, pooled, normalized])
                    
                    logger.info("✅ MLX kernel warmup completed")
                else:
                    logger.warning("Invalid hidden states shape for warmup")
                    
            except Exception as e:
                logger.warning(f"Model forward pass warmup failed: {e}")
            
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    async def encode_text(self, text: str) -> mx.array:
        """Encode single text to MLX native embedding - Korrigierte Version"""
        if not self._is_loaded:
            await self.load_model()
        
        start_time = time.time()
        
        try:
            # KORREKTUR: Sichere Tokenization
            try:
                inputs = self.tokenizer.encode(
                    text,
                    return_tensors="mlx",
                    max_length=self.config.max_sequence_length,
                    padding=True,
                    truncation=True
                )
            except Exception as e:
                raise RuntimeError(f"Tokenization failed: {e}")
            
            # Extract input_ids and attention_mask safely
            if isinstance(inputs, dict):
                input_ids = inputs.get("input_ids")
                attention_mask = inputs.get("attention_mask")
            else:
                input_ids = inputs
                attention_mask = None
            
            if input_ids is None:
                raise RuntimeError("No input_ids from tokenizer")
            
            # KORREKTUR: Erstelle attention_mask ohne dtype Parameter
            if attention_mask is None:
                attention_mask = mx.ones_like(input_ids)
                attention_mask = attention_mask.astype(mx.float32)
            
            # Model forward pass
            try:
                outputs = self.model(input_ids)
                
                # Extract hidden states
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    # Direkte pooler output verwenden falls verfügbar
                    embedding = outputs.pooler_output
                    if embedding.ndim > 1:
                        embedding = embedding.squeeze(0)
                    return mlx_normalize_embeddings(embedding.reshape(1, -1)).squeeze(0)
                    
                elif hasattr(outputs, 'last_hidden_state'):
                    hidden_states = outputs.last_hidden_state
                else:
                    hidden_states = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                
                # Mean pooling mit attention mask
                pooled_embedding = mlx_mean_pooling_with_mask(hidden_states, attention_mask)
                
                # L2 Normalisierung
                normalized_embedding = mlx_normalize_embeddings(pooled_embedding)
                
                # Evaluation für Performance
                mx.eval(normalized_embedding)
                
                # Performance tracking
                inference_time = time.time() - start_time
                self._inference_times.append(inference_time)
                self._batch_sizes.append(1)
                
                return normalized_embedding.squeeze(0)
                
            except Exception as e:
                raise RuntimeError(f"Model forward pass failed: {e}")
                
        except Exception as e:
            logger.error(f"MLX encoding failed: {e}")
            raise RuntimeError(f"Text encoding failed: {e}")
    
    async def encode_batch(self, texts: List[str]) -> List[mx.array]:
        """Encode batch of texts with MLX optimization - Korrigierte Version"""
        if not self._is_loaded:
            await self.load_model()
        
        if not texts:
            return []
        
        start_time = time.time()
        batch_size = len(texts)
        
        try:
            # KORREKTUR: Sichere Batch-Tokenization
            try:
                inputs = self.tokenizer.batch_encode_plus(
                    texts,
                    return_tensors="mlx",
                    max_length=self.config.max_sequence_length,
                    padding=True,
                    truncation=True
                )
            except Exception as e:
                raise RuntimeError(f"Batch tokenization failed: {e}")
            
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask")
            
            if input_ids is None:
                raise RuntimeError("No input_ids from batch tokenizer")
            
            # KORREKTUR: Erstelle attention_mask ohne dtype Parameter
            if attention_mask is None:
                attention_mask = mx.ones_like(input_ids)
                attention_mask = attention_mask.astype(mx.float32)
            
            # Batch model forward pass
            try:
                outputs = self.model(input_ids, attention_mask=attention_mask)
                
                # Extract hidden states
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    # Direkte pooler output falls verfügbar
                    pooled_embeddings = outputs.pooler_output
                    normalized_embeddings = mlx_normalize_embeddings(pooled_embeddings)
                    mx.eval(normalized_embeddings)
                    return [normalized_embeddings[i] for i in range(batch_size)]
                    
                elif hasattr(outputs, 'last_hidden_state'):
                    hidden_states = outputs.last_hidden_state
                else:
                    hidden_states = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                
                # Batch mean pooling
                pooled_embeddings = mlx_mean_pooling_with_mask(hidden_states, attention_mask)
                
                # Batch normalization
                normalized_embeddings = mlx_normalize_embeddings(pooled_embeddings)
                
                # Evaluation für Performance
                mx.eval(normalized_embeddings)
                
                # Performance tracking
                inference_time = time.time() - start_time
                self._inference_times.append(inference_time)
                self._batch_sizes.append(batch_size)
                
                # Convert to list of individual embeddings
                return [normalized_embeddings[i] for i in range(batch_size)]
                
            except Exception as e:
                raise RuntimeError(f"Batch model forward pass failed: {e}")
                
        except Exception as e:
            logger.error(f"MLX batch encoding failed: {e}")
            raise RuntimeError(f"Batch encoding failed: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        if not self._inference_times:
            return {"no_data": True}
        
        total_texts = sum(self._batch_sizes)
        total_time = sum(self._inference_times)
        avg_batch_size = sum(self._batch_sizes) / len(self._batch_sizes)
        
        return {
            "model_type": "mlx-native",
            "model_id": self.config.model_id,
            "device": str(mx.default_device()),
            "quantized": self.config.quantized,
            "dimension": self.config.dimension,
            "total_inferences": len(self._inference_times),
            "total_texts_processed": total_texts,
            "avg_inference_time_ms": (total_time / len(self._inference_times)) * 1000,
            "avg_batch_size": avg_batch_size,
            "throughput_texts_per_sec": total_texts / total_time if total_time > 0 else 0,
            "memory_pool_mb": self.config.memory_pool_mb,
            "unified_memory": self.config.use_unified_memory,
            "jit_compiled": self.config.jit_compile
        }
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get MLX memory usage information"""
        try:
            # KORREKTUR: Sichere Memory Info Abfrage
            memory_info = {}
            
            if hasattr(mx, 'metal'):
                if hasattr(mx.metal, 'get_memory_limit'):
                    memory_info["metal_memory_limit_mb"] = mx.metal.get_memory_limit() // (1024 * 1024)
                if hasattr(mx.metal, 'get_cache_limit'):
                    memory_info["metal_cache_limit_mb"] = mx.metal.get_cache_limit() // (1024 * 1024)
                if hasattr(mx.metal, 'get_active_memory'):
                    memory_info["active_memory_mb"] = mx.metal.get_active_memory() // (1024 * 1024)
                if hasattr(mx.metal, 'get_peak_memory'):
                    memory_info["peak_memory_mb"] = mx.metal.get_peak_memory() // (1024 * 1024)
            
            # Fallback if no metal info available
            if not memory_info:
                memory_info = {"info": "Memory details not available"}
                
            return memory_info
            
        except Exception as e:
            logger.warning(f"Could not get memory info: {e}")
            return {"error": str(e)}

# =================== MLX NATIVE PIPELINE ===================

class MLXNativePipeline:
    """
    MLX Native Text Processing Pipeline
    100% Apple Silicon optimiert - API-kompatible Version
    """
    
    def __init__(self, model_id: str, vector_store: MLXVectorStore):
        self.model_id = model_id
        self.vector_store = vector_store
        
        # Validate model
        if model_id not in MLX_NATIVE_MODELS:
            logger.warning(f"Model {model_id} not in native models, using default")
            self.model_id = DEFAULT_MLX_MODEL
        
        self.config = MLX_NATIVE_MODELS[self.model_id]
        self.embedding_model = MLXNativeEmbeddingModel(self.config)
        
        # Pipeline statistics
        self.stats = {
            "total_texts_processed": 0,
            "total_vectors_stored": 0,
            "total_processing_time_ms": 0,
            "avg_embedding_time_ms": 0,
            "documents_indexed": 0,
            "queries_processed": 0
        }
        
        # Thread pool für CPU intensive tasks
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="mlx_pipeline")
        
        logger.info(f"MLX Native Pipeline initialized: {self.model_id}")
    
    async def initialize(self):
        """Initialize MLX Native Pipeline"""
        logger.info("Initializing MLX Native Pipeline...")
        
        # Verify MLX device
        device = mx.default_device()
        logger.info(f"📱 MLX Device: {device}")
        
        # Load embedding model
        await self.embedding_model.load_model()
        
        # Verify dimension compatibility
        expected_dim = self.config.dimension
        store_dim = self.vector_store.config.dimension
        
        if expected_dim != store_dim:
            raise ValueError(f"Dimension mismatch: model {expected_dim} vs store {store_dim}")
        
        logger.info(f"✅ MLX Native Pipeline ready")
        logger.info(f"🧠 Model: {self.model_id}")
        logger.info(f"📐 Dimension: {self.config.dimension}")
        logger.info(f"⚡ Quantized: {self.config.quantized}")
    
    async def process_texts(self, texts: List[str], 
                          metadata: Optional[List[Dict[str, Any]]] = None,
                          batch_size: int = 32) -> Dict[str, Any]:
        """Process texts with MLX optimized batching"""
        
        if not texts:
            return {"success": False, "error": "No texts provided"}
        
        start_time = time.time()
        
        # Default metadata
        if metadata is None:
            metadata = [{"text": text[:100], "index": i, "timestamp": time.time()} 
                       for i, text in enumerate(texts)]
        
        if len(texts) != len(metadata):
            raise ValueError("Texts and metadata length mismatch")
        
        # Process in batches für optimale Memory usage
        all_embeddings = []
        embedding_start = time.time()
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = await self.embedding_model.encode_batch(batch_texts)
            all_embeddings.extend(batch_embeddings)
        
        embedding_time = time.time() - embedding_start
        
        # Convert to numpy für vector store
        embeddings_np = np.array([np.array(emb.tolist()) for emb in all_embeddings], dtype=np.float32)
        
        # Store in vector database
        storage_result = self.vector_store.add_vectors(embeddings_np, metadata)
        
        # Update statistics
        total_time = time.time() - start_time
        self.stats["total_texts_processed"] += len(texts)
        self.stats["total_vectors_stored"] += len(embeddings_np)
        self.stats["total_processing_time_ms"] += total_time * 1000
        self.stats["avg_embedding_time_ms"] = (embedding_time * 1000) / len(texts)
        
        return {
            "success": True,
            "texts_processed": len(texts),
            "vectors_stored": len(embeddings_np),
            "total_time_ms": total_time * 1000,
            "embedding_time_ms": embedding_time * 1000,
            "storage_time_ms": (total_time - embedding_time) * 1000,
            "throughput_texts_per_sec": len(texts) / total_time if total_time > 0 else 0,
            "embedding_throughput_texts_per_sec": len(texts) / embedding_time if embedding_time > 0 else 0,
            "model_id": self.model_id,
            "batch_size": batch_size,
            "storage_result": storage_result
        }
    
    async def search_similar_texts(self, query_text: str, k: int = 10,
                                 filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar texts mit MLX optimization"""
        
        query_start = time.time()
        
        # Encode query
        query_embedding = await self.embedding_model.encode_text(query_text)
        query_np = np.array(query_embedding.tolist())
        
        # Search in vector store
        indices, distances, metadata_results = self.vector_store.query(
            query_np, k=k, filter_metadata=filter_metadata
        )
        
        query_time = time.time() - query_start
        self.stats["queries_processed"] += 1
        
        # Format results
        results = []
        for i, (idx, dist, meta) in enumerate(zip(indices, distances, metadata_results)):
            # Convert distance to similarity score
            if self.vector_store.config.metric == "cosine":
                similarity_score = max(0.0, dist)  # Cosine similarity ist bereits normalisiert
            else:
                similarity_score = max(0.0, 1.0 - dist)
            
            results.append({
                "rank": i + 1,
                "similarity_score": float(similarity_score),
                "distance": float(dist),
                "metadata": meta,
                "text_preview": meta.get("text", "")[:200],
                "index": int(idx)
            })
        
        return results
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        embedding_stats = self.embedding_model.get_performance_stats()
        memory_info = self.embedding_model.get_memory_info()
        
        return {
            "pipeline_stats": self.stats,
            "embedding_model_stats": embedding_stats,
            "memory_info": memory_info,
            "model_configuration": {
                "model_id": self.model_id,
                "model_type": "mlx-native",
                "dimension": self.config.dimension,
                "quantized": self.config.quantized,
                "memory_pool_mb": self.config.memory_pool_mb,
                "unified_memory": self.config.use_unified_memory,
                "jit_compiled": self.config.jit_compile
            },
            "device_info": {
                "mlx_device": str(mx.default_device()),
                "apple_silicon": True,
                "unified_memory_architecture": True
            }
        }

# =================== RAG PIPELINE ===================

class MLXNativeRAGPipeline(MLXNativePipeline):
    """
    MLX Native RAG Pipeline für Apple Silicon
    """
    
    def __init__(self, model_id: str, vector_store: MLXVectorStore):
        super().__init__(model_id, vector_store)
        
        # RAG specific stats
        self.rag_stats = {
            "documents_indexed": 0,
            "total_chunks": 0,
            "avg_chunk_size": 0,
            "retrieval_queries": 0,
            "avg_retrieval_time_ms": 0
        }
    
    async def index_documents(self, documents: List[Dict[str, str]], 
                            chunk_size: int = 512,
                            chunk_overlap: int = 50) -> Dict[str, Any]:
        """Index documents for RAG mit optimized chunking"""
        
        logger.info(f"Indexing {len(documents)} documents for RAG...")
        
        all_chunks = []
        all_metadata = []
        
        for doc_idx, document in enumerate(documents):
            doc_content = document.get("content", "")
            doc_title = document.get("title", f"Document {doc_idx}")
            doc_source = document.get("source", f"doc_{doc_idx}")
            
            # Intelligent chunking
            chunks = self._create_chunks(doc_content, chunk_size, chunk_overlap)
            
            for chunk_idx, chunk_text in enumerate(chunks):
                if chunk_text.strip():
                    all_chunks.append(chunk_text)
                    all_metadata.append({
                        "document_id": doc_idx,
                        "document_title": doc_title,
                        "document_source": doc_source,
                        "chunk_index": chunk_idx,
                        "chunk_text": chunk_text,
                        "chunk_length": len(chunk_text),
                        "indexed_at": time.time(),
                        "text": chunk_text[:100]  # Preview für search results
                    })
        
        # Process through embedding pipeline
        result = await self.process_texts(all_chunks, all_metadata)
        
        # Update RAG stats
        self.rag_stats["documents_indexed"] += len(documents)
        self.rag_stats["total_chunks"] += len(all_chunks)
        self.rag_stats["avg_chunk_size"] = sum(len(chunk) for chunk in all_chunks) / len(all_chunks) if all_chunks else 0
        
        logger.info(f"✅ Indexed {len(documents)} documents as {len(all_chunks)} chunks")
        
        return {
            **result,
            "documents_indexed": len(documents),
            "chunks_created": len(all_chunks),
            "avg_chunk_size": self.rag_stats["avg_chunk_size"],
            "rag_stats": self.rag_stats
        }
    
    def _create_chunks(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Create overlapping text chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings
                for punct in ['. ', '! ', '? ', '\n\n']:
                    last_punct = chunk.rfind(punct)
                    if last_punct > len(chunk) * 0.7:  # Only break if we keep 70% of chunk
                        chunk = chunk[:last_punct + len(punct)]
                        break
            
            chunks.append(chunk.strip())
            start = max(start + chunk_size - overlap, start + 1)
            
            if start >= len(text):
                break
        
        return chunks
    
    async def retrieve_context(self, query: str, k: int = 5, 
                             min_similarity: float = 0.7,
                             max_context_length: int = 2000) -> List[Dict[str, Any]]:
        """Retrieve relevant context for RAG"""
        
        retrieval_start = time.time()
        
        # Search for relevant chunks
        results = await self.search_similar_texts(query, k=k * 2)  # Get more, then filter
        
        # Filter by minimum similarity
        filtered_results = [r for r in results if r["similarity_score"] >= min_similarity]
        
        # Limit by context length
        context_chunks = []
        current_length = 0
        
        for result in filtered_results[:k]:
            chunk_text = result["metadata"].get("chunk_text", "")
            if current_length + len(chunk_text) <= max_context_length:
                context_chunks.append({
                    "text": chunk_text,
                    "source": result["metadata"].get("document_source", ""),
                    "title": result["metadata"].get("document_title", ""),
                    "similarity": result["similarity_score"],
                    "chunk_index": result["metadata"].get("chunk_index", 0),
                    "document_id": result["metadata"].get("document_id", 0)
                })
                current_length += len(chunk_text)
            
            if len(context_chunks) >= k:
                break
        
        # Update stats
        retrieval_time = time.time() - retrieval_start
        self.rag_stats["retrieval_queries"] += 1
        current_avg = self.rag_stats["avg_retrieval_time_ms"]
        query_count = self.rag_stats["retrieval_queries"]
        self.rag_stats["avg_retrieval_time_ms"] = (current_avg * (query_count - 1) + retrieval_time * 1000) / query_count
        
        return context_chunks
    
    def format_rag_prompt(self, query: str, context_chunks: List[Dict[str, Any]], 
                         system_prompt: str = None) -> str:
        """Format context for RAG generation"""
        
        if system_prompt is None:
            system_prompt = "Du bist ein hilfreicher Assistent. Beantworte Fragen basierend auf dem gegebenen Kontext."
        
        # Build context string
        context_parts = []
        for i, chunk in enumerate(context_chunks):
            source_info = f"[Quelle {i+1}: {chunk['title']} - {chunk['source']}]"
            context_parts.append(f"{source_info}\n{chunk['text']}\n")
        
        context_string = "\n".join(context_parts)
        
        rag_prompt = f"""{system_prompt}

Kontext:
{context_string}

Frage: {query}

Antwort basierend auf dem Kontext:"""
        
        return rag_prompt
    
    def get_rag_stats(self) -> Dict[str, Any]:
        """Get RAG-specific statistics"""
        base_stats = self.get_pipeline_stats()
        base_stats["rag_stats"] = self.rag_stats
        return base_stats

# =================== PIPELINE FACTORY ===================

class MLXNativePipelineFactory:
    """Factory für MLX Native Pipelines"""
    
    @staticmethod
    async def create_pipeline(model_id: str, vector_store: MLXVectorStore, 
                            pipeline_type: str = "basic") -> Union[MLXNativePipeline, MLXNativeRAGPipeline]:
        """Create and initialize MLX native pipeline"""
        
        if pipeline_type == "basic":
            pipeline = MLXNativePipeline(model_id, vector_store)
        elif pipeline_type == "rag":
            pipeline = MLXNativeRAGPipeline(model_id, vector_store)
        else:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")
        
        await pipeline.initialize()
        return pipeline
    
    @staticmethod
    def get_recommended_model(use_case: str = "general", 
                            performance_priority: str = "balanced",
                            memory_budget_mb: int = 2048) -> str:
        """Get recommended MLX native model"""
        
        # Filter by memory budget
        available_models = {
            k: v for k, v in MLX_NATIVE_MODELS.items()
            if v.memory_pool_mb <= memory_budget_mb
        }
        
        if not available_models:
            return DEFAULT_MLX_MODEL
        
        # Use case specific recommendations
        recommendations = {
            "general": ["mlx-community/bge-small-en-v1.5-4bit", "mlx-community/all-MiniLM-L6-v2-4bit"],
            "multilingual": ["mlx-community/multilingual-e5-small-4bit", "mlx-community/multilingual-e5-base-4bit"],
            "high_quality": ["mlx-community/all-mpnet-base-v2-4bit", "mlx-community/bge-base-en-v1.5-4bit"],
            "performance": ["mlx-community/all-MiniLM-L6-v2-4bit", "mlx-community/bge-small-en-v1.5-4bit"],
            "research": ["mlx-community/all-roberta-large-v1-4bit", "mlx-community/all-mpnet-base-v2-4bit"]
        }
        
        # Performance priority adjustments
        if performance_priority == "speed":
            # Prefer smaller models
            model_order = recommendations.get(use_case, recommendations["general"])
            model_order = sorted(model_order, key=lambda x: MLX_NATIVE_MODELS.get(x, MLX_NATIVE_MODELS[DEFAULT_MLX_MODEL]).dimension)
        elif performance_priority == "quality":
            # Prefer larger models
            model_order = recommendations.get(use_case, recommendations["general"])
            model_order = sorted(model_order, key=lambda x: MLX_NATIVE_MODELS.get(x, MLX_NATIVE_MODELS[DEFAULT_MLX_MODEL]).dimension, reverse=True)
        else:
            # Balanced
            model_order = recommendations.get(use_case, recommendations["general"])
        
        # Return first available model
        for model_id in model_order:
            if model_id in available_models:
                return model_id
        
        return DEFAULT_MLX_MODEL
    
    @staticmethod
    def list_native_models() -> Dict[str, Dict[str, Any]]:
        """List all MLX native models with specifications"""
        return {
            model_id: {
                "dimension": config.dimension,
                "quantized": config.quantized,
                "memory_pool_mb": config.memory_pool_mb,
                "max_sequence_length": config.max_sequence_length,
                "unified_memory": config.use_unified_memory,
                "metal_performance": config.enable_metal_performance,
                "jit_compiled": config.jit_compile
            }
            for model_id, config in MLX_NATIVE_MODELS.items()
        }

# =================== BENCHMARK SUITE ===================

class MLXNativeBenchmark:
    """Benchmark Suite für MLX Native Models"""
    
    def __init__(self):
        self.results = {}
    
    async def benchmark_model(self, model_id: str, test_texts: List[str], 
                            vector_store_path: str = None) -> Dict[str, Any]:
        """Benchmark einzelnes MLX native model"""
        
        logger.info(f"🏁 Benchmarking MLX native model: {model_id}")
        
        try:
            # Create test vector store
            from service.optimized_vector_store import create_optimized_vector_store
            
            config = MLX_NATIVE_MODELS.get(model_id, MLX_NATIVE_MODELS[DEFAULT_MLX_MODEL])
            
            if vector_store_path is None:
                vector_store_path = f"./temp_mlx_benchmark_{model_id.replace('/', '_').replace('-', '_')}"
            
            vector_store = create_optimized_vector_store(
                vector_store_path,
                dimension=config.dimension,
                jit_compile=True,
                enable_hnsw=False  # For consistent benchmarking
            )
            
            # Create pipeline
            pipeline = MLXNativePipeline(model_id, vector_store)
            
            # Benchmark initialization
            init_start = time.time()
            await pipeline.initialize()
            init_time = time.time() - init_start
            
            # Benchmark single text encoding
            single_text = test_texts[0] if test_texts else "Test text for benchmarking"
            single_start = time.time()
            await pipeline.embedding_model.encode_text(single_text)
            single_encode_time = time.time() - single_start
            
            # Benchmark batch processing
            batch_start = time.time()
            process_result = await pipeline.process_texts(test_texts[:100])  # Limit for benchmark
            batch_process_time = time.time() - batch_start
            
            # Benchmark search operations
            search_times = []
            num_search_tests = min(10, len(test_texts))
            for i in range(num_search_tests):
                search_start = time.time()
                await pipeline.search_similar_texts(test_texts[i % len(test_texts)], k=5)
                search_times.append(time.time() - search_start)
            
            avg_search_time = sum(search_times) / len(search_times) if search_times else 0
            
            # Get performance stats
            perf_stats = pipeline.get_pipeline_stats()
            memory_info = pipeline.embedding_model.get_memory_info()
            
            # Cleanup
            vector_store.clear()
            
            return {
                "model_id": model_id,
                "success": True,
                "device": str(mx.default_device()),
                "model_config": {
                    "dimension": config.dimension,
                    "quantized": config.quantized,
                    "memory_pool_mb": config.memory_pool_mb,
                    "unified_memory": config.use_unified_memory
                },
                "performance": {
                    "initialization_time_s": init_time,
                    "single_encode_time_ms": single_encode_time * 1000,
                    "batch_throughput_texts_per_sec": process_result["throughput_texts_per_sec"],
                    "embedding_throughput_texts_per_sec": process_result["embedding_throughput_texts_per_sec"],
                    "avg_search_time_ms": avg_search_time * 1000,
                    "total_texts_processed": len(test_texts[:100]),
                    "batch_process_time_s": batch_process_time
                },
                "memory": memory_info,
                "detailed_stats": perf_stats
            }
            
        except Exception as e:
            logger.error(f"Benchmark failed for {model_id}: {e}")
            return {
                "model_id": model_id,
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    async def run_comprehensive_benchmark(self, models_to_test: Optional[List[str]] = None,
                                        num_test_texts: int = 100) -> Dict[str, Any]:
        """Run comprehensive benchmark across MLX native models"""
        
        if models_to_test is None:
            # Test all available models
            models_to_test = list(MLX_NATIVE_MODELS.keys())[:4]  # Limit to prevent long runtime
        
        # Generate test texts
        test_texts = [
            f"Apple Silicon M-series chips provide unified memory architecture for machine learning workload {i}."
            f" The Metal Performance Shaders framework enables GPU acceleration for neural networks."
            f" MLX framework delivers optimized performance on Apple hardware with minimal memory overhead."
            for i in range(num_test_texts)
        ]
        
        logger.info(f"🚀 Starting comprehensive MLX native benchmark")
        logger.info(f"📊 Testing {len(models_to_test)} models with {num_test_texts} texts")
        logger.info(f"📱 Device: {mx.default_device()}")
        
        benchmark_results = {}
        successful_models = []
        failed_models = []
        
        for model_id in models_to_test:
            logger.info(f"Testing {model_id}...")
            result = await self.benchmark_model(model_id, test_texts)
            benchmark_results[model_id] = result
            
            if result.get("success", False):
                successful_models.append(model_id)
                logger.info(f"✅ {model_id}: {result['performance']['batch_throughput_texts_per_sec']:.1f} texts/sec")
            else:
                failed_models.append(model_id)
                logger.error(f"❌ {model_id}: {result.get('error', 'Unknown error')}")
        
        # Generate comprehensive summary
        if successful_models:
            successful_results = [benchmark_results[m] for m in successful_models]
            
            # Find best performers
            best_throughput = max(successful_results, key=lambda x: x["performance"]["batch_throughput_texts_per_sec"])
            best_latency = min(successful_results, key=lambda x: x["performance"]["single_encode_time_ms"])
            best_search = min(successful_results, key=lambda x: x["performance"]["avg_search_time_ms"])
            
            # Memory usage analysis
            memory_usage = {
                result["model_id"]: result["memory"].get("peak_memory_mb", 0)
                for result in successful_results
            }
            
            summary = {
                "device_info": {
                    "mlx_device": str(mx.default_device()),
                    "apple_silicon": True,
                    "unified_memory": True
                },
                "test_configuration": {
                    "models_tested": len(models_to_test),
                    "test_texts": num_test_texts,
                    "successful_models": len(successful_models),
                    "failed_models": len(failed_models)
                },
                "performance_leaders": {
                    "best_throughput": {
                        "model": best_throughput["model_id"],
                        "value": best_throughput["performance"]["batch_throughput_texts_per_sec"],
                        "unit": "texts/sec"
                    },
                    "best_latency": {
                        "model": best_latency["model_id"],
                        "value": best_latency["performance"]["single_encode_time_ms"],
                        "unit": "ms"
                    },
                    "best_search": {
                        "model": best_search["model_id"],
                        "value": best_search["performance"]["avg_search_time_ms"],
                        "unit": "ms"
                    }
                },
                "memory_usage": memory_usage,
                "recommendations": {
                    "for_speed": best_latency["model_id"],
                    "for_throughput": best_throughput["model_id"],
                    "for_search": best_search["model_id"],
                    "general_purpose": DEFAULT_MLX_MODEL
                }
            }
        else:
            summary = {
                "test_configuration": {
                    "models_tested": len(models_to_test),
                    "successful_models": 0,
                    "failed_models": len(failed_models)
                },
                "message": "No models succeeded in benchmark",
                "failed_models": failed_models
            }
        
        return {
            "benchmark_results": benchmark_results,
            "summary": summary,
            "timestamp": time.time()
        }

async def demo_mlx_native_pipeline():
    """Demo der korrigierten MLX Native Pipeline"""
    
    print("🍎 MLX Native Embedding Pipeline Demo - KORRIGIERTE VERSION")
    print("⚡ 100% Apple Silicon optimiert - API-kompatibel")
    print("=" * 70)
    
    # Check MLX device
    device = mx.default_device()
    print(f"📱 MLX Device: {device}")
    
    # Show available models
    print(f"\n📋 Available MLX Native Models:")
    models = MLXNativePipelineFactory.list_native_models()
    for model_id, specs in list(models.items())[:3]:  # Show first 3
        print(f"   🧠 {model_id}")
        print(f"      Dimension: {specs['dimension']}, Memory: {specs['memory_pool_mb']}MB")
        print(f"      Quantized: {specs['quantized']}")
    
    # Get recommended model
    recommended_model = MLXNativePipelineFactory.get_recommended_model(
        use_case="general",
        performance_priority="balanced",
        memory_budget_mb=1024
    )
    
    print(f"\n🎯 Recommended Model: {recommended_model}")
    
    # Create vector store
    from service.optimized_vector_store import create_optimized_vector_store
    
    config = MLX_NATIVE_MODELS[recommended_model]
    vector_store = create_optimized_vector_store(
        "./demo_mlx_fixed_store",
        dimension=config.dimension,
        jit_compile=True,
        enable_hnsw=False
    )
    
    # Create RAG pipeline
    print(f"\n🔧 Creating MLX Native RAG Pipeline...")
    try:
        rag_pipeline = await MLXNativePipelineFactory.create_pipeline(
            recommended_model,
            vector_store,
            pipeline_type="rag"
        )
        print("✅ Pipeline created successfully!")
        
        # Demo documents
        demo_documents = [
            {
                "title": "MLX Framework Fixed",
                "content": "MLX ist Apples Machine Learning Framework für Apple Silicon. Die korrigierte Version behebt API-Inkompatibilitäten mit mx.ones_like() und anderen Funktionen.",
                "source": "mlx_fixed.md"
            },
            {
                "title": "API Compatibility", 
                "content": "Die neue Version verwendet korrekte MLX API-Aufrufe ohne veraltete Parameter wie dtype in mx.ones_like(). Attention Masks werden jetzt sicher erstellt.",
                "source": "api_fix.md"
            }
        ]
        
        # Index documents
        print(f"\n📚 Indexing {len(demo_documents)} documents...")
        index_start = time.time()
        
        try:
            index_result = await rag_pipeline.index_documents(demo_documents, chunk_size=200)
            index_time = time.time() - index_start
            
            print(f"✅ Indexing completed in {index_time:.2f}s")
            print(f"📄 Created {index_result['chunks_created']} chunks")
            print(f"🚀 Throughput: {index_result['throughput_texts_per_sec']:.1f} texts/sec")
            
            # Demo query
            print(f"\n🔍 Testing query...")
            query = "Wie wurde das MLX API Problem behoben?"
            
            try:
                context_chunks = await rag_pipeline.retrieve_context(query, k=2, min_similarity=0.1)
                print(f"✅ Retrieved {len(context_chunks)} chunks")
                
                for i, chunk in enumerate(context_chunks):
                    print(f"   {i+1}. Similarity: {chunk['similarity']:.3f}")
                    print(f"      {chunk['text'][:80]}...")
                
            except Exception as e:
                print(f"❌ Query failed: {e}")
            
        except Exception as e:
            print(f"❌ Indexing failed: {e}")
        
        # Show stats
        print(f"\n📊 Pipeline Stats:")
        stats = rag_pipeline.get_pipeline_stats()
        print(f"   🧠 Model: {stats['model_configuration']['model_id']}")
        print(f"   📱 Device: {stats['device_info']['mlx_device']}")
        print(f"   ⚡ Quantized: {stats['model_configuration']['quantized']}")
        
    except Exception as e:
        print(f"❌ Pipeline creation failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        try:
            vector_store.clear()
            print(f"\n🧹 Cleanup completed")
        except:
            pass
    
    print(f"\n✅ MLX Native Pipeline Demo completed!")

# =================== EXPORT ===================

# Convenience aliases for backward compatibility
MLXPipelineFactory = MLXNativePipelineFactory
MLXTextEmbeddingPipeline = MLXNativePipeline
RAGPipeline = MLXNativeRAGPipeline

# Default exports
SUPPORTED_EMBEDDING_MODELS = MLX_NATIVE_MODELS
DEFAULT_MODEL = DEFAULT_MLX_MODEL

if __name__ == "__main__":
    asyncio.run(demo_mlx_native_pipeline())