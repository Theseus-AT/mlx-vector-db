#!/usr/bin/env python3
"""
Copyright (c) 2024 Theseus AI Technologies. All rights reserved.

This software is proprietary and confidential. Commercial use requires 
a valid enterprise license from Theseus AI Technologies.

Enterprise Contact: enterprise@theseus-ai.com
Website: https://theseus-ai.com/enterprise
"""

# benchmark_app.py (FINALE VERSION MIT SYNTAX-FIX)
import sys
import time
import asyncio
from pathlib import Path

# === ANFANG: ROBUSTER IMPORT-FIX ===
try:
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
except NameError:
    project_root = Path.cwd()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
# === ENDE: ROBUSTER IMPORT-FIX ===

import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import httpx
import shutil

# --- Modul-Imports ---
try:
    from sdk.python.mlx_vector_client import MLXVectorClient
    from service.production_integration import ProductionVectorStoreManager
    import faiss
    import chromadb
    from qdrant_client import QdrantClient, models
    import mlx.core as mx
except ImportError as e:
    print(f"Fehler beim Importieren der Module: {e}")
    print("Bitte führe `pip install mlx faiss-cpu chromadb qdrant-client pandas matplotlib` aus.")
    exit()

# === Globale Konfiguration und Clients ===
API_SERVER_URL = "http://localhost:8000"
API_KEY = "mlx-vector-dev-key-2024"
mlx_api_client = MLXVectorClient(base_url=API_SERVER_URL, api_key=API_KEY)

# === Einheitliche Benchmark-Funktion (mit korrigierter Struktur) ===
async def run_full_benchmark(engines, num_vectors, dimension, num_queries, k, progress=gr.Progress(track_tqdm=True)):
    results_data = []
    dimension = int(dimension)

    for engine in progress.tqdm(engines, desc="Testing Engines..."):
        ingestion_vps, query_qps, avg_latency_ms, error = 0, 0, 0, None
        store_id, collection_name, direct_benchmark_path = None, None, None # Für Cleanup

        # KORREKTUR: Der try...except...finally Block umschließt jetzt die gesamte Operation
        try:
            vectors = np.random.rand(num_vectors, dimension).astype("float32")
            ingestion_start_time = time.time()

            # --- Setup & Ingestion ---
            if engine == "MLX DB (API)":
                store_id = f"api-test-{int(time.time())}"
                await mlx_api_client.create_store("bench", store_id, dimension=dimension)
                await mlx_api_client.add_vectors("bench", store_id, vectors.tolist(), [{"id": i} for i in range(num_vectors)])

            elif engine == "MLX DB (Direkt)":
                direct_benchmark_path = project_root / f"temp_direct_benchmark_stores_{int(time.time())}"
                manager = ProductionVectorStoreManager(base_path=str(direct_benchmark_path))
                store_id = "direct-test"
                from service.optimized_vector_store import MLXVectorStoreConfig
                config = MLXVectorStoreConfig(dimension=dimension)
                await manager.create_store("bench", store_id, config=config)
                await manager.add_vectors("bench", store_id, vectors, [{"id": i} for i in range(num_vectors)])
                mx.synchronize()

            elif engine == "FAISS":
                index = faiss.IndexFlatL2(dimension)
                index.add(vectors)

            elif engine == "ChromaDB":
                client = chromadb.Client()
                collection_name = f"chroma-test-{int(time.time())}"
                collection = client.create_collection(name=collection_name)
                batch_size = 4096
                for i in range(0, num_vectors, batch_size):
                    end_index = min(i + batch_size, num_vectors)
                    collection.add(embeddings=vectors[i:end_index].tolist(), ids=[str(j) for j in range(i, end_index)])

            elif engine == "Qdrant":
                client = QdrantClient(":memory:")
                collection_name = f"qdrant-test-{int(time.time())}"
                client.recreate_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE)
                )
                client.upsert(
                    collection_name=collection_name,
                    points=[models.PointStruct(id=i, vector=v.tolist()) for i, v in enumerate(vectors)],
                    wait=True
                )

            ingestion_time = time.time() - ingestion_start_time
            ingestion_vps = num_vectors / ingestion_time if ingestion_time > 0 else float('inf')

            # --- Querying ---
            query_vectors = np.random.rand(num_queries, dimension).astype("float32")
            latencies = []
            query_total_start_time = time.time()
            for qv in query_vectors:
                q_start = time.time()
                if engine == "MLX DB (API)":
                    await mlx_api_client.query_vectors("bench", store_id, qv.tolist(), k=k)
                elif engine == "MLX DB (Direkt)":
                    await manager.query_vectors("bench", store_id, qv, k=k)
                elif engine == "FAISS":
                    index.search(np.expand_dims(qv, axis=0), k)
                elif engine == "ChromaDB":
                    collection.query(query_embeddings=[qv.tolist()], n_results=k)
                elif engine == "Qdrant":
                    client.search(collection_name=collection_name, query_vector=qv.tolist(), limit=k)
                latencies.append(time.time() - q_start)
            
            query_total_time = time.time() - query_total_start_time
            avg_latency_ms = (sum(latencies) / len(latencies)) * 1000 if latencies else 0
            query_qps = num_queries / query_total_time if query_total_time > 0 else float('inf')

        except Exception as e:
            error = f"{type(e).__name__}: {e}"
        
        finally:
            # --- Teardown ---
            # Dieser Block wird immer ausgeführt, auch wenn ein Fehler auftritt.
            try:
                if engine == "MLX DB (API)" and store_id: await mlx_api_client.delete_store("bench", store_id, force=True)
                elif engine == "MLX DB (Direkt)" and direct_benchmark_path: shutil.rmtree(direct_benchmark_path, ignore_errors=True)
                elif engine == "ChromaDB" and 'collection' in locals(): client.delete_collection(name=collection.name)
            except Exception as e:
                print(f"Cleanup-Fehler für {engine}: {e}")
        
        # Das Sammeln der Ergebnisse geschieht NACH dem try-except-finally Block.
        results_data.append({
            "Engine": engine, "Ingestion (Vec/s)": f"{ingestion_vps:,.0f}",
            "Query (QPS)": f"{query_qps:,.0f}", "Avg Latency (ms)": f"{avg_latency_ms:.3f}",
            "Status": "Error" if error else "OK", "Details": error
        })

    # --- Ergebnisse aufbereiten ---
    if not results_data: return pd.DataFrame(), None
    df = pd.DataFrame(results_data)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    df_numeric = df.copy()
    for col in ["Ingestion (Vec/s)", "Query (QPS)"]:
        df_numeric[col] = pd.to_numeric(df_numeric[col].str.replace(',', ''), errors='coerce')
    df_numeric.plot(kind='bar', x='Engine', y='Ingestion (Vec/s)', ax=ax[0], color='skyblue', legend=False, title='Ingestion (höher ist besser)')
    ax[0].set_ylabel('Vektoren / Sekunde')
    ax[0].tick_params(axis='x', rotation=45)
    df_numeric.plot(kind='bar', x='Engine', y='Query (QPS)', ax=ax[1], color='lightgreen', legend=False, title='Query (höher ist besser)')
    ax[1].set_ylabel('Queries / Sekunde (QPS)')
    ax[1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    return df, fig

def create_advanced_app():
    with gr.Blocks(theme=gr.themes.Monochrome(), title="Vector DB Comparison Hub") as app:
        gr.Markdown("# 🍎 Vector DB Performance & Comparison Hub")
        # ... UI-Code bleibt unverändert ...
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Benchmark-Parameter")
                engines_to_test = gr.CheckboxGroup(choices=["MLX DB (API)", "MLX DB (Direkt)", "FAISS", "ChromaDB", "Qdrant"], value=["MLX DB (API)", "MLX DB (Direkt)", "FAISS"], label="Zu testende Engines")
                num_vectors = gr.Slider(minimum=1000, maximum=50000, value=10000, step=1000, label="Anzahl Vektoren")
                dimension = gr.Dropdown(choices=[128, 384, 768], value=384, label="Vektor-Dimension", type="value")
                num_queries = gr.Slider(minimum=100, maximum=2000, value=500, step=100, label="Anzahl Queries")
                k = gr.Slider(minimum=1, maximum=100, value=10, step=1, label="k (Top-N)")
                run_btn = gr.Button("Vollständigen Benchmark starten", variant="primary", scale=2)
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Ergebnisse")
                results_df = gr.DataFrame(label="Vergleichsergebnisse", wrap=True)
                results_plot = gr.Plot(label="Performance-Vergleich")
        run_btn.click(fn=run_full_benchmark, inputs=[engines_to_test, num_vectors, dimension, num_queries, k], outputs=[results_df, results_plot])
    return app

if __name__ == "__main__":
    advanced_app = create_advanced_app()
    advanced_app.launch()