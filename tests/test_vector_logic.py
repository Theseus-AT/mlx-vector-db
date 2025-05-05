import numpy as np
import pytest
from service.vector_store import (
    create_store,
    store_exists,
    delete_store,
    add_vectors,
    count_vectors,
    query_vectors,
    delete_vectors,
)

user_id = "logic_test_user"
model = "logic_test_model"

def setup_module(module):
    if store_exists(user_id, model):
        delete_store(user_id, model)
    create_store(user_id, model)

def teardown_module(module):
    if store_exists(user_id, model):
        delete_store(user_id, model)

def test_add_and_count_consistency():
    vecs = np.random.rand(3, 384).astype(np.float32)
    meta = [{"id": f"x{i}"} for i in range(3)]
    add_vectors(user_id, model, vecs, meta)
    stats = count_vectors(user_id, model)
    assert stats["vectors"] == 3
    assert stats["metadata"] == 3

def test_metadata_mismatch_raises():
    vecs = np.random.rand(2, 384).astype(np.float32)
    meta = [{"id": "x1"}]  # mismatch
    with pytest.raises(ValueError):
        add_vectors(user_id, model, vecs, meta)

def test_invalid_vector_type_raises():
    with pytest.raises(TypeError):
        add_vectors(user_id, model, "not a vector", [])

def test_empty_vector_add_skipped():
    vecs = np.zeros((0, 384), dtype=np.float32)
    add_vectors(user_id, model, vecs, [])
    stats = count_vectors(user_id, model)
    assert stats["vectors"] >= 3  # vorheriger Count bleibt bestehen

def test_delete_by_metadata():
    deleted = delete_vectors(user_id, model, {"id": "x1"})
    assert deleted == 1
    stats = count_vectors(user_id, model)
    assert stats["vectors"] == 2
