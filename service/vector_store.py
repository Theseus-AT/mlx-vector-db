# service/vector_store.py
import shutil
from typing import Optional, List, Dict, Any, Union, Generator
from pathlib import Path
import os
import numpy as np
from filelock import FileLock, Timeout
import json
import logging
import time
import uuid

# --- MLX Imports ---
import mlx.core as mx
import mlx.core.linalg as mxl
# -------------------

# Import utils
try:
    from utils import ensure_directory, get_lock, validate_vector_shape
except ImportError:
    logging.critical("Failed import from utils.py.", exc_info=True)
    def ensure_directory(*args, **kwargs): raise ImportError("utils.py not found")
    def get_lock(*args, **kwargs): raise ImportError("utils.py not found")
    def validate_vector_shape(*args, **kwargs): raise ImportError("utils.py not found")

# Logging setup
logger = logging.getLogger("mlx_vector_db")
if not logger.hasHandlers(): logging.basicConfig(level=logging.DEBUG)
logger.setLevel(logging.DEBUG)

# log_tensor_debug (wie zuvor)
def log_tensor_debug(name: str, tensor) -> None:
    is_mlx = hasattr(tensor, 'shape') and isinstance(tensor, mx.array); is_numpy = isinstance(tensor, np.ndarray)
    if not (is_mlx or is_numpy): logger.debug(f"{name} type: {type(tensor)}"); return
    shape = tensor.shape; size = tensor.size; logger.debug(f"{name}.shape = {shape}")
    if size == 0: logger.debug(f"{name} is empty."); return
    if is_numpy:
        try: logger.debug(f"{name} (numpy).min = {np.min(tensor)}, max = {np.max(tensor)}, mean = {np.mean(tensor)}")
        except Exception as e: logger.error(f"Error numpy stats {name}: {e}")
    elif is_mlx:
        try: min_val=mx.min(tensor).item(); max_val=mx.max(tensor).item(); mean_val=mx.mean(tensor).item(); logger.debug(f"{name} (mlx).min = {min_val}, max = {max_val}, mean = {mean_val}")
        except Exception as e: logger.error(f"Error mlx stats {name}: {e}")
    else: logger.debug(f"{name} (type: {type(tensor)}) - cannot get stats.")


# Global base path
BASE_VECTOR_STORE_PATH = Path.home() / ".team_mind_data" / "vector_stores"

# get_store_path (wie zuvor)
def get_store_path(user_id: str, model_name: str) -> Path:
    safe_user_id = user_id.replace("..", "").replace("/", "_").replace("\\", "_"); safe_model_name = model_name.replace("..", "").replace("/", "_").replace("\\", "_")
    if not safe_user_id or not safe_model_name: raise ValueError("User ID/Model Name invalid.")
    return BASE_VECTOR_STORE_PATH / f"user_{safe_user_id}" / safe_model_name

# create_store (wie zuvor)
def create_store(user_id: str, model_name: str) -> None:
    path = get_store_path(user_id, model_name); logger.debug(f"Ensuring directory: {path}"); ensure_directory(path)
    logger.debug(f"Attempting lock for create: {path}");
    try:
        lock = get_lock(path); logger.debug(f"Acquiring lock: {lock.lock_file}")
        with lock:
            logger.debug(f"Lock acquired create: {path}"); metadata_path = path / "metadata.jsonl"
            if not metadata_path.exists():
                logger.debug(f"Creating metadata: {metadata_path}")
                try: metadata_path.write_text("", encoding='utf-8'); logger.debug("Metadata created.")
                except IOError as e: logger.error(f"Failed create metadata: {e}", exc_info=True); raise
            else: logger.debug("Metadata exists.")
            logger.debug("Store structure prepared.")
        logger.debug(f"Lock released create: {path}")
    except Timeout: logger.error(f"Lock timeout create: {path}", exc_info=True); raise TimeoutError(f"Lock timeout create: {path}") from None
    except Exception as e: logger.exception(f"Unexpected error create: {path}"); raise
    logger.info(f"Store checked/created: {path} for user='{user_id}', model='{model_name}'")

# store_exists (wie zuvor)
def store_exists(user_id: str, model_name: str) -> bool:
     path = get_store_path(user_id, model_name); metadata_path = path / "metadata.jsonl"
     return path.is_dir() and metadata_path.is_file()

# add_vectors (wie zuvor - mit mx.savez und .npz)
def add_vectors(
    user_id: str,
    model_name: str,
    vectors: Union[np.ndarray, "mx.core.Tensor"],
    metadata: List[Dict[str, Any]]
) -> None:
    if not store_exists(user_id, model_name): logger.error(f"Store does not exist: {user_id}/{model_name}. Cannot add."); raise FileNotFoundError(f"Store does not exist: {user_id}/{model_name}.")
    path = get_store_path(user_id, model_name); vector_path = path / "vectors.npz"; metadata_path = path / "metadata.jsonl"
    if not isinstance(vectors, (np.ndarray, mx.array)): raise TypeError(f"Vectors type invalid: {type(vectors)}")
    if not isinstance(metadata, list): raise TypeError(f"Metadata type invalid: {type(metadata)}")
    if len(vectors) == 0: logger.warning(f"Skipping empty vector batch for {user_id}/{model_name}."); return
    if len(vectors) != len(metadata): raise ValueError(f"Vector/Metadata count mismatch: {len(vectors)} != {len(metadata)}")
    if isinstance(vectors, np.ndarray): vectors_np = vectors.astype(np.float32, copy=False); vectors_mx = mx.array(vectors_np)
    else: vectors_mx = vectors;
    if vectors_mx.dtype != mx.float32: logger.warning("Converting input MLX tensor to float32."); vectors_mx = vectors_mx.astype(mx.float32)
    try: validate_vector_shape(vectors_mx, expected_dim=-1)
    except ValueError as e: logger.error(f"Invalid vector shape: {e}", exc_info=True); raise
    embedding_dim = vectors_mx.shape[1]; logger.debug(f"Input vectors shape: {vectors_mx.shape}")
    logger.debug(f"Attempting lock for add_vectors: {path}")
    try:
        lock = get_lock(path); logger.debug(f"Acquiring lock: {lock.lock_file}")
        with lock:
            logger.debug(f"Lock acquired for add_vectors: {path}")
            existing_vectors = None
            if vector_path.exists():
                logger.debug(f"Loading existing vectors from NPZ: {vector_path}")
                try:
                    loaded_data = mx.load(str(vector_path))
                    if 'vectors' not in loaded_data: raise ValueError("Invalid NPZ file: missing 'vectors' key.")
                    existing_vectors = loaded_data['vectors']; log_tensor_debug("existing_vectors (loaded from NPZ)", existing_vectors)
                    if existing_vectors.size > 0:
                        if existing_vectors.ndim != 2: raise ValueError("Corrupted store: Existing vectors not 2D.")
                        if existing_vectors.shape[1] != embedding_dim: raise ValueError(f"Dim mismatch: Store {existing_vectors.shape[1]}, trying {embedding_dim}")
                        if existing_vectors.dtype != mx.float32: logger.warning("Converting loaded vectors to float32."); existing_vectors = existing_vectors.astype(mx.float32)
                    elif existing_vectors.ndim != 2 or existing_vectors.shape[1] != embedding_dim: logger.warning(f"Loaded empty tensor has wrong shape {existing_vectors.shape}, correcting."); existing_vectors = mx.zeros((0, embedding_dim), dtype=mx.float32)
                except Exception as e: logger.exception(f"Failed load existing vectors from NPZ: {vector_path}."); raise
            else: logger.debug(f"Vector file (.npz) not found. Creating empty base: {vector_path}"); existing_vectors = mx.zeros((0, embedding_dim), dtype=mx.float32); log_tensor_debug("existing_vectors (created empty)", existing_vectors)
            log_tensor_debug("new_tensor (to add)", vectors_mx)
            try: combined = mx.concatenate([existing_vectors, vectors_mx], axis=0); log_tensor_debug("combined", combined); mx.eval(combined); logger.debug("Evaluated combined tensor.")
            except Exception as e: logger.exception(f"Error during concatenate: existing={existing_vectors.shape}, new={vectors_mx.shape}."); raise
            vector_save_success = False
            if combined is not None and combined.size > 0:
                logger.debug(f"Attempting vector save using mx.savez: {vector_path} (shape: {combined.shape})")
                try:
                    mx.savez(str(vector_path), vectors=combined); logger.debug(f"mx.savez command issued for {vector_path}.")
                    print("Calling mx.synchronize() after savez..."); start_sync = time.time(); mx.synchronize(); end_sync = time.time(); logger.debug(f"mx.synchronize() completed after {end_sync - start_sync:.4f} seconds.")
                    if not vector_path.exists(): logger.error(f"[ERROR AFTER SYNC W/ SAVEZ] File {vector_path} NOT found!"); raise IOError(f"Failed confirm vector file existence (savez): {vector_path}")
                    else: logger.debug(f"[VERIFY OK AFTER SYNC W/ SAVEZ] File {vector_path} exists."); vector_save_success = True
                except Exception as e: logger.exception(f"[CRITICAL SAVEZ/SYNC] Exception for {vector_path}."); raise
            elif combined is None: logger.error("Cannot save: 'combined' is None."); raise RuntimeError("Combined tensor not created.")
            else: logger.warning("Combined tensor is empty, skipping vector save."); vector_save_success = True
            metadata_save_success = False
            if vector_save_success:
                logger.debug(f"Attempting metadata append: {metadata_path}")
                try:
                    with metadata_path.open("a", encoding="utf-8") as f:
                        for i, entry in enumerate(metadata):
                            if not isinstance(entry, dict): logger.warning(f"Metadata {i} not dict, skipping."); continue
                            try: json_str = json.dumps(entry); f.write(json_str + "\n")
                            except TypeError as json_err: logger.error(f"Metadata {i} not JSON serializable: {entry}. Error: {json_err}.", exc_info=True); continue
                    logger.debug(f"Successfully appended metadata: {metadata_path}"); metadata_save_success = True
                except IOError as e: logger.exception(f"Failed to open/write metadata: {metadata_path}."); logger.critical(f"CRITICAL INCONSISTENCY: Vectors saved ({vector_path}), metadata failed."); raise IOError(f"Failed write metadata. Store inconsistent: {path}") from e
            else: logger.error("Skipping metadata save: vector save failed.")
            if vector_save_success and metadata_save_success: logger.info(f"Successfully added {vectors_mx.shape[0]} vectors/metadata (.npz) to {user_id}/{model_name}")
            else: logger.error(f"Add vectors finished inconsistently. VectorOK={vector_save_success}, MetaOK={metadata_save_success}")
        logger.debug(f"Lock released for add_vectors: {path}")
    except Timeout: logger.error(f"Lock timeout add_vectors: {path}", exc_info=True); raise TimeoutError(f"Lock timeout add_vectors: {path}") from None
    except Exception as e: logger.exception(f"Unexpected error add_vectors: {path}"); raise

# query_vectors (angepasst für NPZ / np.array())
def query_vectors(user_id: str, model_name: str, query_vector: Union[np.ndarray, "mx.core.Tensor"], k: int = 5, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    logger.debug(f"Querying store {user_id}/{model_name} (NPZ format) with k={k}")
    if not store_exists(user_id, model_name): raise FileNotFoundError(f"Store not found: {user_id}/{model_name}")
    path = get_store_path(user_id, model_name); vector_path = path / "vectors.npz"; metadata_path = path / "metadata.jsonl"
    if not vector_path.exists(): logger.warning(f"Vector file (.npz) missing for query: {vector_path}"); return []
    if isinstance(query_vector, np.ndarray): query_mx = mx.array(query_vector.astype(np.float32))
    elif isinstance(query_vector, mx.array): query_mx = query_vector.astype(mx.float32)
    else: raise TypeError(f"Query vector type invalid: {type(query_vector)}")
    if query_mx.ndim == 1: query_mx = query_mx.reshape(1, -1)
    elif not (query_mx.ndim == 2 and query_mx.shape[0] == 1): raise ValueError(f"Query shape invalid: {query_mx.shape}")
    embedding_dim = query_mx.shape[1]; results = []
    try:
        lock = get_lock(path)
        with lock:
            loaded_data = mx.load(str(vector_path))
            if 'vectors' not in loaded_data: raise ValueError(f"Invalid NPZ {vector_path}: missing 'vectors'.")
            db_vectors = loaded_data['vectors']
            if db_vectors.size == 0: return []
            if db_vectors.shape[1] != embedding_dim: raise ValueError(f"Dim mismatch: Query {embedding_dim}, store {db_vectors.shape[1]}")
            with metadata_path.open("r", encoding="utf-8") as f: metadata_lines = [json.loads(line) for line in f]
            if len(metadata_lines) != db_vectors.shape[0]: raise ValueError("Inconsistent store count.")
            original_indices = list(range(db_vectors.shape[0]))
            if filter_metadata:
                indices_to_keep = [i for i, meta in enumerate(metadata_lines) if all(meta.get(key) == value for key, value in filter_metadata.items())]
                if not indices_to_keep: return []
                db_vectors = db_vectors[indices_to_keep]; original_indices = indices_to_keep
                if db_vectors.shape[0] == 0: return []
            norm_db = mxl.norm(db_vectors, axis=1, keepdims=True); norm_query = mxl.norm(query_mx, axis=1, keepdims=True); epsilon = 1e-10
            normalized_db = db_vectors / mx.maximum(norm_db, epsilon); normalized_query = query_mx / mx.maximum(norm_query, epsilon)
            scores = mx.matmul(normalized_query, normalized_db.T); mx.eval(scores)
            scores_np = np.array(scores).flatten() # Verwende np.array()
            num_results = min(k, len(scores_np));
            if num_results <= 0: return []
            top_k_indices_in_filtered = np.argsort(scores_np)[-num_results:][::-1]
            for idx_in_filtered in top_k_indices_in_filtered:
                original_idx = original_indices[idx_in_filtered]; entry = metadata_lines[original_idx].copy()
                entry['similarity_score'] = float(scores_np[idx_in_filtered]); results.append(entry)
    except Timeout: logger.error(f"Lock timeout query: {path}", exc_info=True); raise
    except Exception as e: logger.exception(f"Error during query: {path}"); raise
    logger.info(f"Query {user_id}/{model_name} completed, {len(results)} results.")
    return results

# batch_query (angepasst für NPZ / np.array())
def batch_query(user_id: str, model_name: str, query_vectors: Union[np.ndarray, "mx.core.Tensor"], k: int = 5, filter_metadata: Optional[Dict[str, Any]] = None) -> List[List[Dict[str, Any]]]:
    logger.debug(f"Batch query (NPZ): {user_id}/{model_name}, {len(query_vectors)} queries, k={k}")
    if not store_exists(user_id, model_name): raise FileNotFoundError(f"Store not found: {user_id}/{model_name}")
    path = get_store_path(user_id, model_name); vector_path = path / "vectors.npz"; metadata_path = path / "metadata.jsonl"
    if not vector_path.exists(): logger.warning(f"Vector file missing: {vector_path}"); return [[] for _ in range(len(query_vectors))]
    if isinstance(query_vectors, np.ndarray): query_mx = mx.array(query_vectors.astype(np.float32))
    elif isinstance(query_vectors, mx.array): query_mx = query_vectors.astype(mx.float32)
    else: raise TypeError(f"Query vectors type invalid: {type(query_vectors)}")
    if query_mx.ndim != 2: raise ValueError(f"Batch query vectors must be 2D, got {query_mx.shape}")
    embedding_dim = query_mx.shape[1]; num_queries = query_mx.shape[0];
    if num_queries == 0: return []
    batch_results = []
    try:
        lock = get_lock(path)
        with lock:
            loaded_data = mx.load(str(vector_path))
            if 'vectors' not in loaded_data: raise ValueError(f"Invalid NPZ: {vector_path}")
            db_vectors = loaded_data['vectors']
            if db_vectors.size == 0: return [[] for _ in range(num_queries)]
            if db_vectors.shape[1] != embedding_dim: raise ValueError(f"Dim mismatch: Query {embedding_dim}, store {db_vectors.shape[1]}")
            with metadata_path.open("r", encoding="utf-8") as f: metadata_lines = [json.loads(line) for line in f]
            if len(metadata_lines) != db_vectors.shape[0]: raise ValueError("Inconsistent store count.")
            original_indices = list(range(db_vectors.shape[0]))
            if filter_metadata:
                indices_to_keep = [i for i, meta in enumerate(metadata_lines) if all(meta.get(key) == value for key, value in filter_metadata.items())]
                if not indices_to_keep: return [[] for _ in range(num_queries)]
                db_vectors = db_vectors[indices_to_keep]; original_indices = indices_to_keep
                if db_vectors.shape[0] == 0: return [[] for _ in range(num_queries)]
            norm_db = mxl.norm(db_vectors, axis=1, keepdims=True); norm_query = mxl.norm(query_mx, axis=1, keepdims=True); epsilon = 1e-10
            normalized_db = db_vectors / mx.maximum(norm_db, epsilon); normalized_query = query_mx / mx.maximum(norm_query, epsilon)
            scores = mx.matmul(normalized_query, normalized_db.T); mx.eval(scores)
            scores_np = np.array(scores) # Verwende np.array()
            num_db_vectors_filtered = scores_np.shape[1]; actual_k = min(k, num_db_vectors_filtered);
            if actual_k <= 0: return [[] for _ in range(num_queries)]
            top_k_indices_per_query = np.argsort(scores_np, axis=1)[:, -actual_k:][:, ::-1]
            for i in range(num_queries):
                query_results = []; top_k_indices = top_k_indices_per_query[i]; scores_k = scores_np[i, top_k_indices]
                for j, idx_in_filtered in enumerate(top_k_indices):
                    original_idx = original_indices[idx_in_filtered]; entry = metadata_lines[original_idx].copy()
                    entry['similarity_score'] = float(scores_k[j]); query_results.append(entry)
                batch_results.append(query_results)
    except Timeout: logger.error(f"Lock timeout batch query: {path}", exc_info=True); raise
    except Exception as e: logger.exception(f"Error batch query: {path}"); raise
    logger.info(f"Batch query {user_id}/{model_name} completed.")
    return batch_results

# =============================================================================
# KORRIGIERTE stream_query function
# =============================================================================
# Korrigierte stream_query Funktion

def stream_query(user_id: str, model_name: str, query_vectors: Union[np.ndarray, "mx.core.Tensor"], k: int = 5, filter_metadata: Optional[Dict[str, Any]] = None) -> Generator[List[Dict[str, Any]], None, None]:
    logger.debug(f"Stream query (NPZ): {user_id}/{model_name}, k={k}")
    if not store_exists(user_id, model_name): raise FileNotFoundError(f"Store not found: {user_id}/{model_name}")
    path = get_store_path(user_id, model_name); vector_path = path / "vectors.npz"; metadata_path = path / "metadata.jsonl"

    # Korrekter Check am Anfang
    if not vector_path.exists():
        logger.warning(f"Vector file (.npz) missing for stream query: {vector_path}")
        num_queries_est = 0
        if hasattr(query_vectors, '__len__'):
            try: num_queries_est = len(query_vectors)
            except TypeError: pass
        for _ in range(num_queries_est):
            yield []
        return # Exit generator

    # Input Validierung und Vorbereitung (wie zuvor)
    if isinstance(query_vectors, np.ndarray): query_mx = mx.array(query_vectors.astype(np.float32))
    elif isinstance(query_vectors, mx.array): query_mx = query_vectors.astype(mx.float32)
    else: raise TypeError(f"Query vectors type invalid: {type(query_vectors)}")
    if query_mx.ndim != 2: raise ValueError(f"Stream query vectors must be 2D, got {query_mx.shape}")
    embedding_dim = query_mx.shape[1]; num_queries = query_mx.shape[0];
    if num_queries == 0: return # Keine Queries -> nichts zu tun

    try:
        lock = get_lock(path)
        with lock:
            # Daten laden
            loaded_data = mx.load(str(vector_path))
            if 'vectors' not in loaded_data: raise ValueError(f"Invalid NPZ: {vector_path}")
            db_vectors = loaded_data['vectors']

            # *** KORREKTUR 1: Check nach Laden ***
            if db_vectors.size == 0:
                logger.info(f"Stream query on empty vector store {user_id}/{model_name}.")
                for _ in range(num_queries):
                    yield []
                return # Exit generator

            if db_vectors.shape[1] != embedding_dim: raise ValueError(f"Dim mismatch: Query {embedding_dim}, store {db_vectors.shape[1]}")

            with metadata_path.open("r", encoding="utf-8") as f: metadata_lines = [json.loads(line) for line in f]
            if len(metadata_lines) != db_vectors.shape[0]: raise ValueError("Inconsistent store count.")

            original_indices = list(range(db_vectors.shape[0]))

            # Filtern
            if filter_metadata:
                indices_to_keep = [i for i, meta in enumerate(metadata_lines) if all(meta.get(key) == value for key, value in filter_metadata.items())]

                # *** KORREKTUR 2: Check nach Filter (indices) ***
                if not indices_to_keep:
                    logger.info(f"No vectors matched metadata filter {filter_metadata} for stream query.")
                    for _ in range(num_queries):
                         yield []
                    return # Exit generator

                db_vectors = db_vectors[indices_to_keep]; original_indices = indices_to_keep

                # *** KORREKTUR 3: Check nach Filter (shape) ***
                if db_vectors.shape[0] == 0:
                    # Sollte durch vorherigen Check abgedeckt sein, aber sicher ist sicher
                    logger.info("Filtered vectors result in empty set for stream query.")
                    for _ in range(num_queries):
                         yield []
                    return # Exit generator

            # Vorbereitung für Abfrage
            norm_db = mxl.norm(db_vectors, axis=1, keepdims=True); normalized_db = db_vectors / mx.maximum(norm_db, 1e-10)
            num_db_vectors_filtered = normalized_db.shape[0]; actual_k = min(k, num_db_vectors_filtered);

            # *** KORREKTUR 4: Check k ***
            if actual_k <= 0:
                 logger.info(f"Effective k is <= 0 ({actual_k}) for stream query.")
                 for _ in range(num_queries):
                     yield []
                 return # Exit generator

            # Hauptschleife für Queries
            for i in range(num_queries):
                q_vec = query_mx[i:i+1]; norm_q = mxl.norm(q_vec, axis=1, keepdims=True); normalized_q = q_vec / mx.maximum(norm_q, 1e-10)
                scores = mx.matmul(normalized_q, normalized_db.T); mx.eval(scores)
                scores_np = np.array(scores).flatten() # Verwende np.array()
                top_k_indices_in_filtered = np.argsort(scores_np)[-actual_k:][::-1]; final_scores = scores_np[top_k_indices_in_filtered]
                query_results = []
                for j, idx_in_filtered in enumerate(top_k_indices_in_filtered):
                    original_idx = original_indices[idx_in_filtered]; entry = metadata_lines[original_idx].copy()
                    entry['similarity_score'] = float(final_scores[j]); query_results.append(entry)
                yield query_results # Yield das Ergebnis für diese eine Query

    except Timeout: logger.error(f"Lock timeout stream query: {path}", exc_info=True); raise
    except Exception as e: logger.exception(f"Error stream query: {path}"); raise
    logger.info(f"Stream query {user_id}/{model_name} completed.")


# delete_vectors (angepasst für NPZ / synchronize)
def delete_vectors(user_id: str, model_name: str, filter_metadata: Dict[str, Any]) -> int:
    logger.debug(f"Deleting vectors (NPZ): {user_id}/{model_name} matching {filter_metadata}")
    if not store_exists(user_id, model_name): logger.warning(f"Cannot delete, store not found: {user_id}/{model_name}"); return 0
    path = get_store_path(user_id, model_name); vector_path = path / "vectors.npz"; metadata_path = path / "metadata.jsonl"; deleted_count = 0
    if not vector_path.exists(): logger.debug("Vector file (.npz) already gone."); return 0
    try:
        lock = get_lock(path)
        with lock:
            loaded_data = mx.load(str(vector_path))
            if 'vectors' not in loaded_data: raise ValueError(f"Invalid NPZ: {vector_path}")
            vectors = loaded_data['vectors']
            with metadata_path.open("r", encoding="utf-8") as f: metadata_lines = [json.loads(line) for line in f]
            if len(metadata_lines) != vectors.shape[0]: raise ValueError("Inconsistent store before delete.")
            if vectors.shape[0] == 0: return 0
            keep_indices = [i for i, entry in enumerate(metadata_lines) if not all(entry.get(k) == v for k, v in filter_metadata.items())]
            num_to_delete = vectors.shape[0] - len(keep_indices);
            if num_to_delete == 0: logger.info("No vectors matched filter."); return 0
            if keep_indices: new_vectors = vectors[keep_indices]; new_metadata = [metadata_lines[i] for i in keep_indices]
            else: embedding_dim = vectors.shape[1]; new_vectors = mx.zeros((0, embedding_dim), dtype=mx.float32); new_metadata = []
            log_tensor_debug("new_vectors (after deletion)", new_vectors)
            mx.savez(str(vector_path), vectors=new_vectors); logger.debug(f"mx.savez issued delete: {vector_path}")
            mx.synchronize(); logger.debug("Synchronized after delete savez.")
            with metadata_path.open("w", encoding="utf-8") as f:
                for entry in new_metadata: f.write(json.dumps(entry) + "\n")
            deleted_count = num_to_delete
    except Timeout: logger.error(f"Lock timeout delete_vectors: {path}", exc_info=True); raise
    except Exception as e: logger.exception(f"Error delete_vectors: {path}"); raise
    logger.info(f"Deleted {deleted_count} vectors from {user_id}/{model_name} (using .npz).")
    return deleted_count

# count_vectors (angepasst für NPZ)
def count_vectors(user_id: str, model_name: str) -> Dict[str, int]:
    logger.debug(f"Counting (NPZ): {user_id}/{model_name}")
    if not store_exists(user_id, model_name): return {"vectors": 0, "metadata": 0}
    path = get_store_path(user_id, model_name); vector_path = path / "vectors.npz"; metadata_path = path / "metadata.jsonl"
    vector_count = 0; metadata_count = 0; error_occurred = False
    try:
        lock = get_lock(path)
        with lock:
            if vector_path.exists():
                try:
                    loaded_data = mx.load(str(vector_path))
                    if 'vectors' not in loaded_data: logger.error(f"Count FAILED: NPZ missing key 'vectors' {vector_path}"); vector_count = -1; error_occurred = True
                    else: vectors_tensor = loaded_data['vectors']; vector_count = vectors_tensor.shape[0]; logger.debug(f"Count vectors OK (NPZ): {vector_count}")
                except Exception as e: logger.error(f"Count FAILED load NPZ: {e}", exc_info=True); vector_count = -1; error_occurred = True
            else: logger.debug("Count vectors: NPZ file not found, count 0."); vector_count = 0
            try:
                 with metadata_path.open("r", encoding="utf-8") as f: metadata_count = sum(1 for _ in f); logger.debug(f"Count metadata OK: {metadata_count}")
            except Exception as e: logger.error(f"Count metadata FAILED read: {e}", exc_info=True); metadata_count = -1; error_occurred = True
    except Timeout: logger.error(f"Count lock timeout: {path}", exc_info=True); vector_count = -1; metadata_count = -1; error_occurred = True
    except Exception as e: logger.exception(f"Count unexpected error: {path}"); vector_count = -1; metadata_count = -1; error_occurred = True
    if not error_occurred and vector_count >= 0 and metadata_count >= 0 and vector_count != metadata_count:
         logger.warning(f"Inconsistency count: Vectors={vector_count}, Metadata={metadata_count} in {user_id}/{model_name}")
    logger.info(f"Counted vectors={vector_count}, metadata={metadata_count} in {user_id}/{model_name}")
    return {"vectors": vector_count, "metadata": metadata_count}

# --- Unchanged functions ---
def bulk_delete(user_id: str, model_name: str, filter_key: str, filter_value: Any) -> int:
    logger.info(f"Bulk deleting (NPZ): {user_id}/{model_name} where {filter_key}={filter_value}")
    return delete_vectors(user_id, model_name, {filter_key: filter_value})

def delete_store(user_id: str, model_name: str) -> None:
    path = get_store_path(user_id, model_name); logger.debug(f"Attempting delete store: {path}")
    if not path.exists(): logger.warning(f"Attempted delete non-existent store: {path}"); return
    if not path.is_dir(): logger.error(f"Path is not dir: {path}"); raise NotADirectoryError(f"{path}")
    try:
        lock_path = path / ".store.lock"; lock = FileLock(str(lock_path), timeout=5)
        try:
            with lock: logger.warning(f"Lock acquired, deleting dir: {path}"); shutil.rmtree(path)
        except (Timeout, FileNotFoundError):
             logger.warning(f"Lock failed/missing for delete {path}, proceeding.", exc_info=True)
             if path.exists(): shutil.rmtree(path); 
             else: logger.info(f"Dir {path} already gone.")
        logger.info(f"Deleted store: {path}")
    except FileNotFoundError: logger.warning(f"Dir {path} already gone.")
    except Exception as e: logger.exception(f"Failed delete store {path}"); raise

def list_users() -> List[str]:
    users = [];
    if not BASE_VECTOR_STORE_PATH.is_dir(): return []
    try:
        for p in BASE_VECTOR_STORE_PATH.iterdir():
            if p.is_dir() and p.name.startswith("user_"):
                user_id_part = p.name[len("user_"):]
                if user_id_part: users.append(user_id_part)
    except OSError as e: logger.error(f"Error listing users: {e}", exc_info=True)
    return users

def list_models(user_id: str) -> List[str]:
    base_user_path = get_store_path(user_id, "dummy").parent; models = []
    if not base_user_path.is_dir(): return []
    try:
        for p in base_user_path.iterdir():
            if p.is_dir() and not p.name.startswith('.'): models.append(p.name)
    except OSError as e: logger.error(f"Error listing models: {e}", exc_info=True)
    return models