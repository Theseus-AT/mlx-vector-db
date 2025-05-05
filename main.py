from typing import List, Dict, Any, Optional
import numpy as np

# main.py
from fastapi import FastAPI
from api.routes.admin import router as admin_router
from api.routes.vectors import router as vectors_router # <- Hinzufügen

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}

# Registriere die Router
app.include_router(admin_router)
app.include_router(vectors_router) # <- Hinzufügen


# Starte die Anwendung
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    from service.vector_store import (
        create_store, add_vectors, query_vectors,
        delete_vectors, delete_store, store_exists
    )

    user_id = "demo_user"
    model = "mistral"

    # Create the store
    create_store(user_id, model)
    if store_exists(user_id, model):
        print(f"✅ Store created for {user_id}/{model}")

    # Add vectors
    vecs = np.random.rand(3, 384).astype(np.float32)
    meta = [{"id": f"chunk_{i}", "source": "test"} for i in range(3)]
    add_vectors(user_id, model, vecs, meta)
    print("➕ Added 3 vectors.")

    # Query
    qvec = vecs[0]
    results = query_vectors(user_id, model, qvec, k=2)
    print("🔍 Query top-2 results:")
    for res in results:
        print(res)

    # Delete one vector
    delete_vectors(user_id, model, {"id": "chunk_1"})
    print("🗑️  Deleted vector with id 'chunk_1'.")

    # Final cleanup
    delete_store(user_id, model)
    print("🧹 Store deleted.")

    # === Extended tests for vector_store ===
    from vector_store import (
        count_vectors, list_users, list_models,
        batch_query, stream_query, bulk_delete
    )

    # Recreate store and add again for extended testing
    create_store(user_id, model)
    add_vectors(user_id, model, vecs, meta)

    print("📊 Vector Count:", count_vectors(user_id, model))
    print("👥 Users:", list_users())
    print(f"📂 Models for {user_id}:", list_models(user_id))

    # Batch Query
    print("🧠 Batch Query (top-2 each):")
    batch_results = batch_query(user_id, model, vecs, k=2)
    for i, res in enumerate(batch_results):
        print(f"Query {i}: {res}")

    # Stream Query
    print("🌊 Stream Query:")
    for i, stream_res in enumerate(stream_query(user_id, model, vecs, k=2)):
        print(f"Stream {i}: {stream_res}")

    # Bulk delete everything with source 'test'
    bulk_delete(user_id, model, "source", "test")
    print("🧹 Bulk deleted all vectors with source='test'.")

    print("📊 After bulk delete:", count_vectors(user_id, model))

    # Final cleanup
    delete_store(user_id, model)
    print("✅ Final store cleanup complete.")