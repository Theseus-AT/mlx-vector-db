#!/usr/bin/env python3
"""
Copyright (c) 2024 Theseus AI Technologies. All rights reserved.

This software is proprietary and confidential. Commercial use requires 
a valid enterprise license from Theseus AI Technologies.

Enterprise Contact: enterprise@theseus-ai.com
Website: https://theseus-ai.com/enterprise
"""

#
import pytest
import numpy as np
from service.optimized_vector_store import (
    create_store, add_vectors, query_vectors, delete_store,
    count_vectors, store_exists
)

def test_store_lifecycle():
    user_id = "test_user"
    model = "test_model"

    # Cleanup vorher
    if store_exists(user_id, model):
        delete_store(user_id, model)

    create_store(user_id, model)
    assert store_exists(user_id, model)

    vecs = np.random.rand(5, 384).astype(np.float32)
    meta = [{"id": f"v{i}", "source": "test"} for i in range(5)]

    add_vectors(user_id, model, vecs, meta)
    stats = count_vectors(user_id, model)
    assert stats["vectors"] == 5
    assert stats["metadata"] == 5

    results = query_vectors(user_id, model, vecs[0], k=3)
    assert len(results) == 3

    delete_store(user_id, model)
    assert not store_exists(user_id, model)