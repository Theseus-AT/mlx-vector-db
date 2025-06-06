## integrations/mlx_lm_pipeline.py
"""
MLX-LM Integration Pipeline für End-to-End Text Processing
Korrigierte Version basierend auf aktuellen MLX-LM APIs
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import asyncio
import logging
from typing import List, Dict, Any, Optional, Union, AsyncGenerator, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger("mlx_lm_integration")

# MLX-LM Imports mit Fehlerbehandlung
try:
    # Korrigierte Imports basierend auf aktueller mlx-lm Struktur
    from mlx_lm import load, generate
    from mlx_lm.utils import load as load_utils
    MLX_LM_AVAILABLE = True
    logger.info("MLX-LM erfolgreich importiert")
except ImportError as e:
    MLX_LM_AVAILABLE = False
    logger.warning(f"MLX-LM nicht verfügbar: {e}")
    logger.info("Installieren Sie mit: pip install mlx-lm")

# Fallback für Sentence Transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers nicht verfügbar")

from service.optimized_vector_store import MLXVectorStore

# =================== EMBEDDING MODEL CONFIGURATIONS ===================

@dataclass
class EmbeddingModelConfig:
    """Konfiguration für Embedding-Modelle"""
    model_path: str
    model_type: str  # "sentence-transformer", "mlx-lm", "mock"
    dimension: int
    max_sequence_length: int = 512
    batch_size: int = 32
    quantization: Optional[str] = None
    trust_remote_code: bool = False

# Vereinfachte, funktionierende Modell-Konfigurationen
SUPPORTED_EMBEDDING_MODELS = {
    "multilingual-e5-small": EmbeddingModelConfig(
        model_path="intfloat/multilingual-e5-small",
        model_type="sentence-transformer" if SENTENCE_TRANSFORMERS_AVAILABLE else "mock",
        dimension=384,
        max_sequence_length=512,
        batch_size=64
    ),
    "mock-384": EmbeddingModelConfig(
        model_path="mock",
        model_type="mock",
        dimension=384,
        max_sequence_length=512,
        batch_size=32
    )
}

# =================== MOCK EMBEDDING MODEL ===================

class MockEmbeddingModel:
    """Mock Embedding Model für Tests und Demos"""
    
    def __init__(self, config: EmbeddingModelConfig):
        self.config = config
        self._is_loaded = True
        logger.info(f"Mock embedding model initialized (dim={config.dimension})")
    
    async def load_model(self):
        """Mock model loading"""
        logger.info("Mock model 'loaded'")
        return True
    
    async def encode_text(self, text: str) -> mx.array:
        """Encode single text to mock embedding"""
        # Deterministisch basierend auf Text-Hash für Konsistenz
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        seed = int(text_hash[:8], 16) % 2**32
        
        np.random.seed(seed)
        embedding = np.random.normal(0, 1, self.config.dimension).astype(np.float32)
        # Normalisieren
        embedding = embedding / np.linalg.norm(embedding)
        
        return mx.array(embedding)
    
    async def encode_batch(self, texts: List[str]) -> List[mx.array]:
        """Encode batch of texts"""
        results = []
        for text in texts:
            embedding = await self.encode_text(text)
            results.append(embedding)
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Mock performance stats"""
        return {
            "model_type": "mock",
            "dimension": self.config.dimension,
            "total_inferences": 0,
            "avg_inference_time_ms": 1.0
        }

# =================== SENTENCE TRANSFORMER MODEL ===================

class SentenceTransformerModel:
    """Sentence Transformer Wrapper"""
    
    def __init__(self, config: EmbeddingModelConfig):
        self.config = config
        self.model = None
        self._is_loaded = False
        self._inference_times = []
    
    async def load_model(self):
        """Load sentence transformer model"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not available")
        
        try:
            self.model = SentenceTransformer(self.config.model_path)
            self._is_loaded = True
            logger.info(f"Loaded sentence transformer: {self.config.model_path}")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")
            raise
    
    async def encode_text(self, text: str) -> mx.array:
        """Encode single text"""
        results = await self.encode_batch([text])
        return results[0]
    
    async def encode_batch(self, texts: List[str]) -> List[mx.array]:
        """Encode batch of texts"""
        if not self._is_loaded:
            await self.load_model()
        
        start_time = time.time()
        
        try:
            # Use sentence transformer
            embeddings_np = self.model.encode(texts, batch_size=self.config.batch_size)
            
            # Convert to MLX arrays
            embeddings_mlx = [mx.array(emb) for emb in embeddings_np]
            
            # Track performance
            inference_time = time.time() - start_time
            self._inference_times.append(inference_time)
            
            return embeddings_mlx
            
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            raise
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self._inference_times:
            return {"no_data": True}
        
        avg_time = sum(self._inference_times) / len(self._inference_times)
        return {
            "model_type": "sentence-transformer",
            "avg_inference_time_ms": avg_time * 1000,
            "total_inferences": len(self._inference_times)
        }

# =================== MLX EMBEDDING MODEL ===================

class MLXEmbeddingModel:
    """MLX-LM basiertes Embedding Model"""
    
    def __init__(self, config: EmbeddingModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
        self._inference_times = []
    
    async def load_model(self):
        """Load MLX-LM model"""
        if not MLX_LM_AVAILABLE:
            raise ImportError("MLX-LM not available")
        
        try:
            # Verwende korrekte MLX-LM API
            self.model, self.tokenizer = load(self.config.model_path)
            self._is_loaded = True
            logger.info(f"Loaded MLX-LM model: {self.config.model_path}")
        except Exception as e:
            logger.error(f"Failed to load MLX-LM model: {e}")
            raise
    
    async def encode_text(self, text: str) -> mx.array:
        """Encode single text with MLX-LM"""
        results = await self.encode_batch([text])
        return results[0]
    
    async def encode_batch(self, texts: List[str]) -> List[mx.array]:
        """Encode batch with MLX-LM"""
        if not self._is_loaded:
            await self.load_model()
        
        start_time = time.time()
        embeddings = []
        
        try:
            for text in texts:
                # Tokenize
                tokens = self.tokenizer.encode(text)
                if len(tokens) > self.config.max_sequence_length:
                    tokens = tokens[:self.config.max_sequence_length]
                
                # Convert to MLX array
                input_ids = mx.array(tokens)[None, :]
                
                # Get embeddings (vereinfacht)
                with mx.no_grad():
                    # Dies ist vereinfacht - echte Implementation hängt vom Modell ab
                    outputs = self.model(input_ids)
                    if hasattr(outputs, 'last_hidden_state'):
                        hidden_states = outputs.last_hidden_state
                    else:
                        hidden_states = outputs
                    
                    # Mean pooling
                    embedding = mx.mean(hidden_states, axis=1).squeeze(0)
                    
                    # Normalize
                    embedding = embedding / mx.linalg.norm(embedding)
                    embeddings.append(embedding)
            
            # Track performance
            inference_time = time.time() - start_time
            self._inference_times.append(inference_time)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"MLX-LM encoding failed: {e}")
            raise
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self._inference_times:
            return {"no_data": True}
        
        avg_time = sum(self._inference_times) / len(self._inference_times)
        return {
            "model_type": "mlx-lm",
            "avg_inference_time_ms": avg_time * 1000,
            "total_inferences": len(self._inference_times)
        }

# =================== EMBEDDING MODEL FACTORY ===================

def create_embedding_model(model_name: str) -> Union[MockEmbeddingModel, SentenceTransformerModel, MLXEmbeddingModel]:
    """Factory function für Embedding Models"""
    
    if model_name not in SUPPORTED_EMBEDDING_MODELS:
        # Fallback zu Mock-Modell
        logger.warning(f"Unknown model {model_name}, using mock model")
        model_name = "mock-384"
    
    config = SUPPORTED_EMBEDDING_MODELS[model_name]
    
    if config.model_type == "mock":
        return MockEmbeddingModel(config)
    elif config.model_type == "sentence-transformer" and SENTENCE_TRANSFORMERS_AVAILABLE:
        return SentenceTransformerModel(config)
    elif config.model_type == "mlx-lm" and MLX_LM_AVAILABLE:
        return MLXEmbeddingModel(config)
    else:
        # Fallback zu Mock
        logger.warning(f"Model type {config.model_type} not available, using mock")
        config.model_type = "mock"
        return MockEmbeddingModel(config)

# =================== TEXT EMBEDDING PIPELINE ===================

class MLXTextEmbeddingPipeline:
    """Vereinfachte Text-Embedding-Pipeline"""
    
    def __init__(self, embedding_model: str, vector_store: MLXVectorStore):
        self.embedding_model_name = embedding_model
        self.vector_store = vector_store
        self.embedding_model = create_embedding_model(embedding_model)
        
        # Processing stats
        self.stats = {
            "total_texts_processed": 0,
            "total_vectors_stored": 0,
            "total_processing_time_ms": 0
        }
    
    async def initialize(self):
        """Initialize pipeline"""
        logger.info("Initializing MLX Text Embedding Pipeline...")
        await self.embedding_model.load_model()
        
        # Verify dimensions match
        expected_dim = self.embedding_model.config.dimension
        store_dim = self.vector_store.config.dimension
        
        if expected_dim != store_dim:
            raise ValueError(f"Dimension mismatch: model {expected_dim} vs store {store_dim}")
        
        logger.info("Pipeline initialization complete")
    
    async def process_texts(self, texts: List[str], 
                          metadata: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Process texts: embed and store"""
        
        start_time = time.time()
        
        # Default metadata
        if metadata is None:
            metadata = [{"text": text[:100], "index": i} for i, text in enumerate(texts)]
        
        # Validate
        if len(texts) != len(metadata):
            raise ValueError("Texts and metadata length mismatch")
        
        # Embed texts
        embeddings = await self.embedding_model.encode_batch(texts)
        
        # Convert to numpy for storage
        embeddings_np = np.array([np.array(emb.tolist()) for emb in embeddings], dtype=np.float32)
        
        # Store in vector database
        storage_result = self.vector_store.add_vectors(embeddings_np, metadata)
        
        # Update stats
        total_time = time.time() - start_time
        self.stats["total_texts_processed"] += len(texts)
        self.stats["total_vectors_stored"] += len(embeddings_np)
        self.stats["total_processing_time_ms"] += total_time * 1000
        
        return {
            "success": True,
            "texts_processed": len(texts),
            "vectors_stored": len(embeddings_np),
            "total_time_ms": total_time * 1000,
            "throughput_texts_per_sec": len(texts) / total_time if total_time > 0 else 0,
            "storage_result": storage_result
        }
    
    async def search_similar_texts(self, query_text: str, k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar texts"""
        
        # Encode query
        query_embedding = await self.embedding_model.encode_text(query_text)
        query_np = np.array(query_embedding.tolist())
        
        # Search
        indices, distances, metadata_results = self.vector_store.query(query_np, k=k)
        
        # Format results
        results = []
        for i, (idx, dist, meta) in enumerate(zip(indices, distances, metadata_results)):
            similarity_score = max(0, 1.0 - dist) if self.vector_store.config.metric == "cosine" else -dist
            
            results.append({
                "rank": i + 1,
                "similarity_score": float(similarity_score),
                "distance": float(dist),
                "metadata": meta,
                "text_preview": meta.get("text", "")[:200]
            })
        
        return results
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        embedding_stats = self.embedding_model.get_performance_stats()
        
        return {
            "pipeline_stats": self.stats,
            "embedding_model_stats": embedding_stats,
            "model_configuration": {
                "embedding_model": self.embedding_model_name,
                "dimension": self.embedding_model.config.dimension
            }
        }

# =================== RAG PIPELINE ===================

class RAGPipeline(MLXTextEmbeddingPipeline):
    """Simplified RAG Pipeline"""
    
    def __init__(self, embedding_model: str, vector_store: MLXVectorStore):
        super().__init__(embedding_model, vector_store)
        self.knowledge_base_stats = {
            "documents_indexed": 0,
            "total_chunks": 0
        }
    
    async def index_documents(self, documents: List[Dict[str, str]]) -> Dict[str, Any]:
        """Index documents for RAG"""
        
        logger.info(f"Indexing {len(documents)} documents for RAG...")
        
        all_texts = []
        all_metadata = []
        
        for doc_idx, document in enumerate(documents):
            doc_text = document.get("content", "")
            doc_source = document.get("source", f"document_{doc_idx}")
            doc_title = document.get("title", "")
            
            # Simple chunking (in production würde man bessere Strategien verwenden)
            chunk_size = 500
            chunks = [doc_text[i:i+chunk_size] for i in range(0, len(doc_text), chunk_size)]
            
            for chunk_idx, chunk in enumerate(chunks):
                if chunk.strip():
                    all_texts.append(chunk)
                    all_metadata.append({
                        "document_id": doc_idx,
                        "document_source": doc_source,
                        "document_title": doc_title,
                        "chunk_index": chunk_idx,
                        "text": chunk,
                        "indexed_at": time.time()
                    })
        
        # Process through embedding pipeline
        result = await self.process_texts(all_texts, all_metadata)
        
        # Update knowledge base stats
        self.knowledge_base_stats["documents_indexed"] += len(documents)
        self.knowledge_base_stats["total_chunks"] += len(all_texts)
        
        logger.info(f"Successfully indexed {len(documents)} documents as {len(all_texts)} chunks")
        
        return {
            **result,
            "documents_indexed": len(documents),
            "chunks_created": len(all_texts),
            "knowledge_base_stats": self.knowledge_base_stats
        }
    
    async def retrieve_context(self, query: str, k: int = 5, min_similarity: float = 0.7) -> List[Dict[str, Any]]:
        """Retrieve relevant context for RAG"""
        
        results = await self.search_similar_texts(query, k=k * 2)
        
        # Filter by minimum similarity
        filtered_results = [r for r in results if r["similarity_score"] >= min_similarity]
        
        # Take top k
        top_results = filtered_results[:k]
        
        # Format for RAG usage
        context_chunks = []
        for result in top_results:
            chunk_info = {
                "text": result["metadata"].get("text", ""),
                "source": result["metadata"].get("document_source", ""),
                "similarity": result["similarity_score"],
                "chunk_index": result["metadata"].get("chunk_index", 0)
            }
            context_chunks.append(chunk_info)
        
        return context_chunks
    
    def format_rag_prompt(self, query: str, context_chunks: List[Dict[str, Any]], 
                         max_context_length: int = 2000) -> str:
        """Format context for RAG generation"""
        
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
        
        rag_prompt = f"""Based on the following context, please answer the question:

Context:
{context_string}

Question: {query}

Answer:"""
        
        return rag_prompt

# =================== PIPELINE FACTORY ===================

class MLXPipelineFactory:
    """Factory für MLX Pipelines"""
    
    @staticmethod
    def create_embedding_pipeline(model_name: str, vector_store: MLXVectorStore,
                                pipeline_type: str = "basic") -> MLXTextEmbeddingPipeline:
        """Create embedding pipeline"""
        
        if pipeline_type == "basic":
            return MLXTextEmbeddingPipeline(model_name, vector_store)
        elif pipeline_type == "rag":
            return RAGPipeline(model_name, vector_store)
        else:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")
    
    @staticmethod
    def get_recommended_model(use_case: str, memory_budget_gb: float = 8.0) -> str:
        """Get recommended model based on use case"""
        
        if use_case == "multilingual" and SENTENCE_TRANSFORMERS_AVAILABLE:
            return "multilingual-e5-small"
        else:
            return "mock-384"  # Fallback zu Mock-Modell
    
    @staticmethod
    def estimate_memory_usage(model_name: str, batch_size: int = 32) -> Dict[str, float]:
        """Estimate memory usage"""
        
        if model_name == "mock-384":
            return {
                "base_model_gb": 0.001,
                "batch_processing_mb": 1.0,
                "total_estimated_gb": 0.001,
                "quantization": "none"
            }
        elif model_name == "multilingual-e5-small":
            return {
                "base_model_gb": 0.5,
                "batch_processing_mb": batch_size * 0.5,
                "total_estimated_gb": 0.5,
                "quantization": "none"
            }
        else:
            return {
                "base_model_gb": 1.0,
                "batch_processing_mb": batch_size,
                "total_estimated_gb": 1.0,
                "quantization": "none"
            }

# =================== DEMO FUNCTION ===================

async def demo_mlx_lm_integration():
    """Demo der MLX-LM Integration"""
    
    print("🚀 MLX-LM Integration Pipeline Demo")
    print("=" * 50)
    
    print(f"MLX-LM verfügbar: {MLX_LM_AVAILABLE}")
    print(f"Sentence Transformers verfügbar: {SENTENCE_TRANSFORMERS_AVAILABLE}")
    
    # Create vector store
    from service.optimized_vector_store import create_optimized_vector_store
    
    vector_store = create_optimized_vector_store(
        "./demo_mlx_lm_store",
        dimension=384,
        jit_compile=True
    )
    
    # Get recommended model
    recommended_model = MLXPipelineFactory.get_recommended_model(
        use_case="multilingual",
        memory_budget_gb=8.0
    )
    
    print(f"📊 Recommended model: {recommended_model}")
    
    # Create RAG pipeline
    rag_pipeline = MLXPipelineFactory.create_embedding_pipeline(
        recommended_model,
        vector_store,
        pipeline_type="rag"
    )
    
    # Initialize pipeline
    print("\n🔧 Initializing pipeline...")
    await rag_pipeline.initialize()
    
    # Sample documents
    documents = [
        {
            "title": "MLX Framework Overview",
            "content": "MLX is Apple's machine learning framework designed for Apple silicon. It provides efficient array operations and supports automatic differentiation.",
            "source": "mlx_docs.md"
        },
        {
            "title": "Vector Databases Explained", 
            "content": "Vector databases store high-dimensional vectors and enable fast similarity search. They are essential for modern AI applications like RAG.",
            "source": "vector_db_guide.md"
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
    
    query = "How does MLX work?"
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
    print(f"   Embedding model: {stats['model_configuration']['embedding_model']}")
    print(f"   Vector dimension: {stats['model_configuration']['dimension']}")
    
    # Cleanup
    vector_store.clear()
    print(f"\n✅ Demo completed successfully!")

if __name__ == "__main__":
    asyncio.run(demo_mlx_lm_integration())