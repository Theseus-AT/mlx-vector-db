# 🧠 MLXVectorDB für `mlx-langchain-lite`

Diese Vektordatenbank ist ein leichtgewichtiges, lokal laufendes Modul zur Verwaltung und Abfrage von Embeddings im Rahmen von **`mlx-langchain-lite`** – einem vollständig lokalen RAG-System für MLX-kompatible Modelle.

---

## 🧭 Zielsetzung

Diese Datenbank wurde speziell für:
- lokale MLX-Projekte (Apple Silicon & Linux),
- Multi-User-Umgebungen,
- agentenbasierte RAG-Szenarien und
- datenschutzfreundliche KMU-Anwendungen

entwickelt.

---

## 🗂️ Struktur

```bash
mlx_vector_db/
├── storage/
│   ├── user_<id>_<modell>/
│   │   ├── vectors.npz
│   │   ├── metadata.jsonl
│   │   └── index.pkl
├── vector_store.py        # Vektor-Indexierung, Einfügen, Abfragen
├── embedding_engine.py    # MLX-Encoder für Text → Embedding
├── rag_handler.py         # Prompt-Zusammenbau, RAG-Logik
├── batch_dispatcher.py    # Batch-Verarbeitung für parallele Agentenanfragen
├── README.md
└── LICENSE
