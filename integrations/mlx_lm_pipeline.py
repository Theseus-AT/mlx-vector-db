# integrations/mlx_lm_pipeline.py
"""
MLX-LM Integration Pipeline für End-to-End Text Processing
Direkte Integration von Text → Embeddings → Vector Store

🚀 Features:
- Eingebaute MLX-LM Modelle für Embeddings
- Quantization Support für Memory Efficiency  
- Token-efficient Batch Processing
- Model-specific Optimizations
- Streaming Text Processing
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import asyncio
import logging
from typing import List, Dict, Any, Optional, Union, AsyncGenerator, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
from concurrent.futures import ThreadPoolExecutor
import threading

# KORRIGIERT: Logger wird am Anfang initialisiert
logger = logging.getLogger("mlx_lm_integration")

# MLX-LM Imports (wenn verfügbar)
try:
    from mlx_lm_pipeline import load, generate
    from mlx_lm.utils import load_config
    # KORRIGIERT: 'ModelType' ist kein gültiger Export aus mlx_lm.models und wird entfernt.
    # from mlx_lm.models import ModelType 
    MLX_LM_AVAILABLE = True
except ImportError:
    MLX_LM_AVAILABLE = False
    logger.warning("MLX-LM not available. Install with: pip install mlx-lm")

from service.optimized_vector_store import MLXVectorStore

# =================== MODEL CONFIGURATIONS ===================

@dataclass
class EmbeddingModelConfig:
    """Configuration for embedding models"""
    model_path: str
    model_type: str  # "sentence-transformer", "mlx-lm", "custom"
    dimension: int
    max_sequence_length: int = 512
    batch_size: int = 32
    quantization: Optional[str] = None  # "4bit", "8bit", None
    trust_remote_code: bool = False
    device_memory_fraction: float = 0.8

# ... (Rest der Datei bleibt unverändert) ...
# (Fügen Sie hier den Rest Ihrer integrations/mlx_lm_pipeline.py Datei ein)

# Vorkonfigurierte Modelle
SUPPORTED_EMBEDDING_MODELS = {
    "multilingual-e5-small": EmbeddingModelConfig(
        model_path="intfloat/multilingual-e5-small",
        model_type="sentence-transformer",
        dimension=384,
        max_sequence_length=512,
        batch_size=64
    ),
    "gte-large": EmbeddingModelConfig(
        model_path="thenlper/gte-large",
        model_type="sentence-transformer", 
        dimension=1024,
        max_sequence_length=512,
        batch_size=32
    ),
    "e5-mistral-7b": EmbeddingModelConfig(
        model_path="intfloat/e5-mistral-7b-instruct",
        model_type="mlx-lm",
        dimension=4096,
        max_sequence_length=32768,
        batch_size=8,
        quantization="4bit"
    ),
    "bge-m3": EmbeddingModelConfig(
        model_path="BAAI/bge-m3",
        model_type="sentence-transformer",
        dimension=1024,
        max_sequence_length=8192,
        batch_size=16
    )
}

# =================== MLX-LM EMBEDDING MODEL ===================

class MLXEmbeddingModel:
    """MLX-optimized embedding model with quantization support"""
    
    def __init__(self, config: EmbeddingModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._model_lock = threading.Lock()
        self._is_loaded = False
        
        # Performance tracking
        self._inference_times = []
        self._batch_sizes = []
        
    async def load_model(self) -> None:
        """Load model with MLX optimizations"""
        if self._is_loaded:
            return
        
        with self._model_lock:
            if self._is_loaded:
                return
                
            logger.info(f"Loading embedding model: {self.config.model_path}")
            start_time = time.time()
            
            try:
                if self.config.model_type == "mlx-lm" and MLX_LM_AVAILABLE:
                    self.model, self.tokenizer = load(
                        self.config.model_path,
                        adapter_path=None
                    )
                    
                    # Apply quantization if specified
                    if self.config.quantization:
                        self.model = self._apply_quantization(self.model)
                    
                elif self.config.model_type == "sentence-transformer":
                    # Fallback to sentence-transformers with MLX conversion
                    self.model = self._load_sentence_transformer()
                
                else:
                    raise ValueError(f"Unsupported model type: {self.config.model_type}")
                
                load_time = time.time() - start_time
                logger.info(f"Model loaded in {load_time:.2f}s")
                
                # Warmup with dummy input
                await self._warmup_model()
                
                self._is_loaded = True
                
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise
    
    def _apply_quantization(self, model):
        """Apply quantization to reduce memory usage"""
        if self.config.quantization == "4bit":
            # MLX 4-bit quantization
            model = mx.quantize(model, group_size=64, bits=4)
            logger.info("Applied 4-bit quantization")
            
        elif self.config.quantization == "8bit":
            # MLX 8-bit quantization  
            model = mx.quantize(model, group_size=64, bits=8)
            logger.info("Applied 8-bit quantization")
            
        return model
    
    def _load_sentence_transformer(self):
        """Load sentence transformer and convert to MLX"""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Load the model
            model = SentenceTransformer(self.config.model_path)
            
            # Convert to MLX (simplified - would need full conversion logic)
            # For now, we'll use the original model and convert outputs
            logger.info(f"Loaded sentence transformer: {self.config.model_path}")
            return model
            
        except ImportError:
            raise ImportError("sentence-transformers not available. Install with: pip install sentence-transformers")
    
    async def _warmup_model(self) -> None:
        """Warmup model with dummy inputs"""
        logger.info("Warming up embedding model...")
        
        dummy_texts = ["This is a warmup text.", "Another warmup example."]
        
        try:
            await self.encode_batch(dummy_texts)
            logger.info("Model warmup completed")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    async def encode_text(self, text: str) -> mx.array:
        """Encode single text to embedding"""
        return await self.encode_batch([text])[0]
    
    async def encode_batch(self, texts: List[str]) -> List[mx.array]:
        """Encode batch of texts to embeddings"""
        if not self._is_loaded:
            await self.load_model()
        
        start_time = time.time()
        
        try:
            if self.config.model_type == "mlx-lm":
                embeddings = await self._encode_with_mlx_lm(texts)
            else:
                embeddings = await self._encode_with_sentence_transformer(texts)
            
            # Track performance
            inference_time = time.time() - start_time
            self._inference_times.append(inference_time)
            self._batch_sizes.append(len(texts))
            
            # Keep only recent metrics
            if len(self._inference_times) > 100:
                self._inference_times.pop(0)
                self._batch_sizes.pop(0)
            
            logger.debug(f"Encoded {len(texts)} texts in {inference_time:.3f}s")
            return embeddings
            
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            raise
    
    async def _encode_with_mlx_lm(self, texts: List[str]) -> List[mx.array]:
        """Encode texts using MLX-LM model"""
        embeddings = []
        
        for text in texts:
            # Tokenize text
            tokens = self.tokenizer.encode(text)
            
            # Truncate if necessary
            if len(tokens) > self.config.max_sequence_length:
                tokens = tokens[:self.config.max_sequence_length]
            
            # Convert to MLX array
            input_ids = mx.array(tokens)[None, :]  # Add batch dimension
            
            # Get embeddings from model
            with mx.no_grad():
                # Forward pass through embedding layer
                embeddings_output = self.model(input_ids)
                
                # Mean pooling over sequence dimension
                if hasattr(embeddings_output, 'last_hidden_state'):
                    hidden_states = embeddings_output.last_hidden_state
                else:
                    hidden_states = embeddings_output
                
                # Mean pooling
                embedding = mx.mean(hidden_states, axis=1).squeeze(0)
                
                # Normalize
                embedding = embedding / mx.linalg.norm(embedding)
                
                embeddings.append(embedding)
        
        return embeddings
    
    async def _encode_with_sentence_transformer(self, texts: List[str]) -> List[mx.array]:
        """Encode texts using sentence transformer (converted to MLX)"""
        # Use sentence transformer to get embeddings
        embeddings_np = self.model.encode(texts, batch_size=self.config.batch_size)
        
        # Convert to MLX arrays
        embeddings_mlx = [mx.array(emb) for emb in embeddings_np]
        
        return embeddings_mlx
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get model performance statistics"""
        if not self._inference_times:
            return {"no_data": True}
        
        avg_time = sum(self._inference_times) / len(self._inference_times)
        avg_batch_size = sum(self._batch_sizes) / len(self._batch_sizes)
        throughput = avg_batch_size / avg_time if avg_time > 0 else 0
        
        return {
            "avg_inference_time_ms": avg_time * 1000,
            "avg_batch_size": avg_batch_size,
            "throughput_texts_per_sec": throughput,
            "total_inferences": len(self._inference_times),
            "model_type": self.config.model_type,
            "quantization": self.config.quantization
        }

# =================== END-TO-END PIPELINE ===================

class MLXTextEmbeddingPipeline:
    """Complete pipeline: Text → Embeddings → Vector Store"""
    
    def __init__(self, embedding_model: str, vector_store: MLXVectorStoreOptimized):
        self.embedding_model_name = embedding_model
        self.vector_store = vector_store
        
        # Load embedding model config
        if embedding_model in SUPPORTED_EMBEDDING_MODELS:
            self.embedding_config = SUPPORTED_EMBEDDING_MODELS[embedding_model]
        else:
            raise ValueError(f"Unsupported embedding model: {embedding_model}")
        
        # Initialize embedding model
        self.embedding_model = MLXEmbeddingModel(self.embedding_config)
        
        # Processing stats
        self.stats = {
            "total_texts_processed": 0,
            "total_vectors_stored": 0,
            "total_processing_time_ms": 0,
            "embedding_time_ms": 0,
            "storage_time_ms": 0
        }
    
    async def initialize(self) -> None:
        """Initialize the complete pipeline"""
        logger.info("Initializing MLX Text Embedding Pipeline...")
        
        # Load embedding model
        await self.embedding_model.load_model()
        
        # Verify vector store compatibility
        if self.vector_store.config.dimension != self.embedding_config.dimension:
            raise ValueError(
                f"Dimension mismatch: embedding model {self.embedding_config.dimension} "
                f"vs vector store {self.vector_store.config.dimension}"
            )
        
        logger.info("Pipeline initialization complete")
    
    async def process_texts(self, texts: List[str], 
                          metadata: Optional[List[Dict[str, Any]]] = None,
                          batch_size: Optional[int] = None) -> Dict[str, Any]:
        """Process texts end-to-end: embed and store"""
        
        start_time = time.time()
        
        # Validate inputs
        if metadata and len(metadata) != len(texts):
            raise ValueError("Metadata length must match texts length")
        
        # Default metadata if not provided
        if metadata is None:
            metadata = [{"text": text[:100], "index": i} for i, text in enumerate(texts)]
        
        # Determine batch size
        batch_size = batch_size or self.embedding_config.batch_size
        
        # Process in batches
        all_embeddings = []
        embedding_start = time.time()
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = await self.embedding_model.encode_batch(batch_texts)
            all_embeddings.extend(batch_embeddings)
        
        embedding_time = time.time() - embedding_start
        
        # Convert MLX arrays to numpy for storage
        storage_start = time.time()
        embeddings_np = np.array([np.array(emb.tolist()) for emb in all_embeddings], dtype=np.float32)
        
        # Store in vector database
        storage_result = self.vector_store.add_vectors(embeddings_np, metadata)
        storage_time = time.time() - storage_start
        
        # Update stats
        total_time = time.time() - start_time
        self.stats["total_texts_processed"] += len(texts)
        self.stats["total_vectors_stored"] += len(embeddings_np)
        self.stats["total_processing_time_ms"] += total_time * 1000
        self.stats["embedding_time_ms"] += embedding_time * 1000
        self.stats["storage_time_ms"] += storage_time * 1000
        
        return {
            "success": True,
            "texts_processed": len(texts),
            "vectors_stored": len(embeddings_np),
            "embedding_time_ms": embedding_time * 1000,
            "storage_time_ms": storage_time * 1000,
            "total_time_ms": total_time * 1000,
            "throughput_texts_per_sec": len(texts) / total_time if total_time > 0 else 0,
            "storage_result": storage_result
        }
    
    async def search_similar_texts(self, query_text: str, k: int = 10,
                                 filter_metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search for similar texts using semantic search"""
        
        # Encode query text
        query_embedding = await self.embedding_model.encode_text(query_text)
        query_np = np.array(query_embedding.tolist())
        
        # Search in vector store
        indices, distances, metadata_results = self.vector_store.query(
            query_np, k=k, filter_metadata=filter_metadata
        )
        
        # Format results
        results = []
        for i, (idx, dist, meta) in enumerate(zip(indices, distances, metadata_results)):
            similarity_score = max(0, 1.0 - dist) if self.vector_store.config.metric == "cosine" else -dist
            
            result = {
                "rank": i + 1,
                "similarity_score": float(similarity_score),
                "distance": float(dist),
                "metadata": meta,
                "text_preview": meta.get("text", "")[:200] + "..." if len(meta.get("text", "")) > 200 else meta.get("text", "")
            }
            results.append(result)
        
        return results
    
    async def process_streaming_texts(self, text_stream: AsyncGenerator[str, None],
                                    batch_size: int = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Process streaming texts with real-time embeddings"""
        
        batch_size = batch_size or self.embedding_config.batch_size
        current_batch = []
        current_metadata = []
        batch_count = 0
        
        async for text in text_stream:
            current_batch.append(text)
            current_metadata.append({
                "text": text[:100],
                "batch": batch_count,
                "timestamp": time.time()
            })
            
            if len(current_batch) >= batch_size:
                # Process current batch
                result = await self.process_texts(current_batch, current_metadata)
                
                yield {
                    "batch_processed": batch_count,
                    "batch_size": len(current_batch),
                    "result": result
                }
                
                # Reset for next batch
                current_batch = []
                current_metadata = []
                batch_count += 1
        
        # Process remaining texts
        if current_batch:
            result = await self.process_texts(current_batch, current_metadata)
            
            yield {
                "batch_processed": batch_count,
                "batch_size": len(current_batch),
                "result": result,
                "final_batch": True
            }
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        
        embedding_stats = self.embedding_model.get_performance_stats()
        vector_store_stats = self.vector_store.get_comprehensive_stats()
        
        # Calculate average processing time per text
        avg_time_per_text = (
            self.stats["total_processing_time_ms"] / self.stats["total_texts_processed"]
            if self.stats["total_texts_processed"] > 0 else 0
        )
        
        return {
            "pipeline_stats": self.stats,
            "avg_time_per_text_ms": avg_time_per_text,
            "embedding_model_stats": embedding_stats,
            "vector_store_stats": vector_store_stats,
            "model_configuration": {
                "embedding_model": self.embedding_model_name,
                "dimension": self.embedding_config.dimension,
                "max_sequence_length": self.embedding_config.max_sequence_length,
                "quantization": self.embedding_config.quantization
            }
        }

# =================== DOCUMENT PROCESSING UTILITIES ===================

class DocumentProcessor:
    """Advanced document processing with chunking and metadata extraction"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, preserve_sentences: bool = True) -> List[str]:
        """Chunk text into overlapping segments"""
        
        if preserve_sentences:
            # Split by sentences first
            import re
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                # Check if adding this sentence would exceed chunk size
                if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    
                    # Start new chunk with overlap
                    if self.overlap > 0 and len(current_chunk) > self.overlap:
                        current_chunk = current_chunk[-self.overlap:] + " " + sentence
                    else:
                        current_chunk = sentence
                else:
                    current_chunk += " " + sentence if current_chunk else sentence
            
            # Add final chunk
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            return chunks
        
        else:
            # Simple character-based chunking
            chunks = []
            
            for i in range(0, len(text), self.chunk_size - self.overlap):
                chunk = text[i:i + self.chunk_size]
                if chunk.strip():
                    chunks.append(chunk.strip())
            
            return chunks
    
    def extract_metadata(self, text: str, source: str = None) -> Dict[str, Any]:
        """Extract metadata from text"""
        
        metadata = {
            "length": len(text),
            "word_count": len(text.split()),
            "source": source or "unknown"
        }
        
        # Extract basic statistics
        sentences = text.split('.')
        metadata["sentence_count"] = len([s for s in sentences if s.strip()])
        
        # Extract keywords (simple approach)
        words = text.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Ignore short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Top keywords
        top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        metadata["keywords"] = [kw[0] for kw in top_keywords]
        
        return metadata
    
    async def process_document(self, text: str, source: str = None,
                             chunk_size: int = None) -> List[Dict[str, Any]]:
        """Process document into chunks with metadata"""
        
        chunk_size = chunk_size or self.chunk_size
        
        # Chunk the text
        chunks = self.chunk_text(text)
        
        # Generate metadata for each chunk
        chunk_data = []
        for i, chunk in enumerate(chunks):
            metadata = self.extract_metadata(chunk, source)
            metadata.update({
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_text": chunk  # Include full text in metadata
            })
            
            chunk_data.append({
                "text": chunk,
                "metadata": metadata
            })
        
        return chunk_data

# =================== SPECIALIZED PIPELINES ===================

class RAGPipeline(MLXTextEmbeddingPipeline):
    """Specialized pipeline for RAG (Retrieval-Augmented Generation)"""
    
    def __init__(self, embedding_model: str, vector_store: MLXVectorStoreOptimized,
                 document_processor: DocumentProcessor = None):
        super().__init__(embedding_model, vector_store)
        
        self.document_processor = document_processor or DocumentProcessor()
        self.knowledge_base_stats = {
            "documents_indexed": 0,
            "total_chunks": 0,
            "total_tokens": 0
        }
    
    async def index_documents(self, documents: List[Dict[str, str]],
                            chunk_size: int = 512) -> Dict[str, Any]:
        """Index documents for RAG retrieval"""
        
        logger.info(f"Indexing {len(documents)} documents for RAG...")
        
        all_chunks = []
        all_metadata = []
        
        for doc_idx, document in enumerate(documents):
            doc_text = document.get("content", "")
            doc_source = document.get("source", f"document_{doc_idx}")
            
            # Process document into chunks
            chunk_data = await self.document_processor.process_document(
                doc_text, doc_source, chunk_size
            )
            
            # Collect chunks and metadata
            for chunk_info in chunk_data:
                all_chunks.append(chunk_info["text"])
                
                # Enhanced metadata for RAG
                metadata = chunk_info["metadata"]
                metadata.update({
                    "document_id": doc_idx,
                    "document_source": doc_source,
                    "document_title": document.get("title", ""),
                    "indexed_at": time.time()
                })
                
                all_metadata.append(metadata)
        
        # Process all chunks through embedding pipeline
        result = await self.process_texts(all_chunks, all_metadata)
        
        # Update knowledge base stats
        self.knowledge_base_stats["documents_indexed"] += len(documents)
        self.knowledge_base_stats["total_chunks"] += len(all_chunks)
        self.knowledge_base_stats["total_tokens"] += sum(len(chunk.split()) for chunk in all_chunks)
        
        logger.info(f"Successfully indexed {len(documents)} documents as {len(all_chunks)} chunks")
        
        return {
            **result,
            "documents_indexed": len(documents),
            "chunks_created": len(all_chunks),
            "knowledge_base_stats": self.knowledge_base_stats
        }
    
    async def retrieve_context(self, query: str, k: int = 5,
                             min_similarity: float = 0.7) -> List[Dict[str, Any]]:
        """Retrieve relevant context for RAG generation"""
        
        # Get similar chunks
        results = await self.search_similar_texts(query, k=k * 2)  # Get more candidates
        
        # Filter by minimum similarity
        filtered_results = [
            result for result in results 
            if result["similarity_score"] >= min_similarity
        ]
        
        # Take top k results
        top_results = filtered_results[:k]
        
        # Format for RAG usage
        context_chunks = []
        for result in top_results:
            chunk_info = {
                "text": result["metadata"].get("chunk_text", ""),
                "source": result["metadata"].get("document_source", ""),
                "similarity": result["similarity_score"],
                "chunk_index": result["metadata"].get("chunk_index", 0)
            }
            context_chunks.append(chunk_info)
        
        return context_chunks
    
    def format_rag_prompt(self, query: str, context_chunks: List[Dict[str, Any]],
                         max_context_length: int = 2000) -> str:
        """Format context for RAG generation"""
        
        # Build context string
        context_parts = []
        current_length = 0
        
        for chunk in context_chunks:
            chunk_text = chunk["text"]
            chunk_with_source = f"[Source: {chunk['source']}]\n{chunk_text}\n"
            
            if current_length + len(chunk_with_source) > max_context_length:
                break
            
            context_parts.append(chunk_with_source)
            current_length += len(chunk_with_source)
        
        context_string = "\n".join(context_parts)
        
        # Create RAG prompt
        rag_prompt = f"""Based on the following context, please answer the question:

Context:
{context_string}

Question: {query}

Answer:"""
        
        return rag_prompt

# =================== PIPELINE FACTORY ===================

class MLXPipelineFactory:
    """Factory for creating optimized MLX pipelines"""
    
    @staticmethod
    def create_embedding_pipeline(model_name: str, vector_store: MLXVectorStoreOptimized,
                                pipeline_type: str = "basic") -> MLXTextEmbeddingPipeline:
        """Create embedding pipeline with specified configuration"""
        
        if pipeline_type == "basic":
            return MLXTextEmbeddingPipeline(model_name, vector_store)
        
        elif pipeline_type == "rag":
            document_processor = DocumentProcessor(
                chunk_size=512,
                overlap=50
            )
            return RAGPipeline(model_name, vector_store, document_processor)
        
        else:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")
    
    @staticmethod
    def get_recommended_model(use_case: str, memory_budget_gb: float = 8.0) -> str:
        """Get recommended model based on use case and memory budget"""
        
        if use_case == "multilingual":
            if memory_budget_gb >= 16:
                return "bge-m3"  # Best multilingual performance
            else:
                return "multilingual-e5-small"  # Memory efficient
        
        elif use_case == "long_documents":
            if memory_budget_gb >= 32:
                return "e5-mistral-7b"  # Best for long context
            else:
                return "gte-large"  # Good balance
        
        elif use_case == "fast_inference":
            return "multilingual-e5-small"  # Fastest
        
        elif use_case == "high_quality":
            if memory_budget_gb >= 16:
                return "bge-m3"  # Best quality
            else:
                return "gte-large"  # Good quality, lower memory
        
        else:
            return "multilingual-e5-small"  # Safe default
    
    @staticmethod
    def estimate_memory_usage(model_name: str, batch_size: int = 32) -> Dict[str, float]:
        """Estimate memory usage for model and batch size"""
        
        if model_name not in SUPPORTED_EMBEDDING_MODELS:
            return {"error": "Unknown model"}
        
        config = SUPPORTED_EMBEDDING_MODELS[model_name]
        
        # Base model memory (rough estimates)
        base_memory_gb = {
            "multilingual-e5-small": 0.5,
            "gte-large": 2.5,
            "e5-mistral-7b": 14.0,
            "bge-m3": 2.2
        }.get(model_name, 1.0)
        
        # Batch processing memory
        sequence_memory_mb = (batch_size * config.max_sequence_length * 4) / (1024 * 1024)  # float32
        
        # Quantization reduction
        if config.quantization == "4bit":
            base_memory_gb *= 0.25
        elif config.quantization == "8bit":
            base_memory_gb *= 0.5
        
        total_memory_gb = base_memory_gb + (sequence_memory_mb / 1024)
        
        return {
            "base_model_gb": base_memory_gb,
            "batch_processing_mb": sequence_memory_mb,
            "total_estimated_gb": total_memory_gb,
            "quantization": config.quantization or "none"
        }

# =================== DEMO USAGE ===================

async def demo_mlx_lm_integration():
    """Demo der MLX-LM Integration"""
    
    print("🚀 MLX-LM Integration Pipeline Demo")
    print("=" * 50)
    
    # Check MLX-LM availability
    if not MLX_LM_AVAILABLE:
        print("⚠️ MLX-LM not available. Using sentence-transformers fallback.")
    
    # Create vector store
    from service.optimized_vector_store import create_optimized_vector_store
    
    vector_store = create_optimized_vector_store(
        "./demo_mlx_lm_store",
        dimension=384,  # For multilingual-e5-small
        jit_compile=True
    )
    
    # Get recommended model
    recommended_model = MLXPipelineFactory.get_recommended_model(
        use_case="multilingual",
        memory_budget_gb=8.0
    )
    
    print(f"📊 Recommended model: {recommended_model}")
    
    # Estimate memory usage
    memory_estimate = MLXPipelineFactory.estimate_memory_usage(
        recommended_model, batch_size=32
    )
    print(f"💾 Estimated memory usage: {memory_estimate['total_estimated_gb']:.2f} GB")
    
    # Create RAG pipeline
    rag_pipeline = MLXPipelineFactory.create_embedding_pipeline(
        recommended_model,
        vector_store,
        pipeline_type="rag"
    )
    
    # Initialize pipeline
    print("\n🔧 Initializing pipeline...")
    await rag_pipeline.initialize()
    
    # Sample documents for indexing
    documents = [
        {
            "title": "MLX Framework Overview",
            "content": "MLX is Apple's machine learning framework designed for Apple silicon. It provides efficient array operations and supports automatic differentiation.",
            "source": "mlx_docs.md"
        },
        {
            "title": "Vector Databases Explained", 
            "content": "Vector databases store high-dimensional vectors and enable fast similarity search. They are essential for modern AI applications like RAG and semantic search.",
            "source": "vector_db_guide.md"
        },
        {
            "title": "Apple Silicon Performance",
            "content": "Apple Silicon chips feature unified memory architecture that allows efficient sharing between CPU and GPU. This enables new optimization strategies for ML workloads.",
            "source": "apple_silicon.md"
        }
    ]
    
    # Index documents
    print(f"\n📚 Indexing {len(documents)} documents...")
    index_result = await rag_pipeline.index_documents(documents)
    
    print(f"✅ Indexed {index_result['documents_indexed']} documents")
    print(f"   Created {index_result['chunks_created']} chunks")
    print(f"   Processing time: {index_result['total_time_ms']:.1f}ms")
    
    # Test RAG retrieval
    print(f"\n🔍 Testing RAG retrieval...")
    
    query = "How does Apple Silicon benefit machine learning?"
    context_chunks = await rag_pipeline.retrieve_context(query, k=3)
    
    print(f"Query: {query}")
    print(f"Retrieved {len(context_chunks)} relevant chunks:")
    
    for i, chunk in enumerate(context_chunks):
        print(f"  {i+1}. [{chunk['source']}] Similarity: {chunk['similarity']:.3f}")
        print(f"     {chunk['text'][:100]}...")
    
    # Format RAG prompt
    rag_prompt = rag_pipeline.format_rag_prompt(query, context_chunks)
    print(f"\n📝 Generated RAG prompt length: {len(rag_prompt)} characters")
    
    # Show pipeline statistics
    print(f"\n📊 Pipeline Statistics:")
    stats = rag_pipeline.get_pipeline_stats()
    
    print(f"   Texts processed: {stats['pipeline_stats']['total_texts_processed']}")
    print(f"   Avg time per text: {stats['avg_time_per_text_ms']:.2f}ms")
    print(f"   Embedding model: {stats['model_configuration']['embedding_model']}")
    print(f"   Vector dimension: {stats['model_configuration']['dimension']}")
    
    # Cleanup
    vector_store.clear()
    print(f"\n✅ Demo completed successfully!")

if __name__ == "__main__":
    asyncio.run(demo_mlx_lm_integration())