# 🧠 MLX Vector Database für Apple Silicon

**High-Performance Vector Database optimiert für Apple Silicon mit MLX 0.25.2**

MLXVectorDB ist eine native Apple Silicon Vector Database, die das MLX Machine Learning Framework nutzt für maximale Performance auf M-Series Chips. Entwickelt für lokale RAG-Systeme, Multi-User-Umgebungen und datenschutzfreundliche AI-Anwendungen.

---

## 🌟 Features

### **🚀 Apple Silicon Optimiert**
* **MLX 0.25.2**: Native Apple Machine Learning Framework
* **Unified Memory**: Zero-copy Operationen zwischen CPU/GPU
* **Metal Kernels**: GPU-Beschleunigung mit Kernel-Caching
* **Lazy Evaluation**: Arrays werden nur bei Bedarf materialisiert
* **ARM64 Native**: Maximale Hardware-Ausnutzung

### **⚡ High-Performance**
* **108,000+ vectors/sec**: Vector Addition Rate
* **112+ QPS**: Query Performance (aktuell, optimierbar auf 1000+ QPS)
* **Sub-10ms Latency**: Einzelne Queries
* **Batch Processing**: Effiziente Multi-Query Verarbeitung
* **MLX Compilation**: JIT-optimierte Operationen

### **🏗️ Enterprise-Ready**
* **FastAPI REST API**: Async, OpenAPI dokumentiert
* **Multi-Store Support**: Separate Stores für User/Modelle
* **Authentication**: API-Key basierte Sicherheit
* **Monitoring**: Prometheus-kompatible Metriken
* **Import/Export**: ZIP-basiertes Backup/Restore

### **🔒 Privatsphäre & Kontrolle**
* **100% Lokal**: Alle Daten bleiben auf dem System
* **Keine Cloud-Abhängigkeit**: Komplett offline betreibbar
* **Metadaten-Filterung**: Präzise Suchkriterien
* **Multi-Tenant**: Sichere User-Isolation

---

## 📊 Performance Benchmarks

**Aktuelle Performance (MLX 0.25.2 Basis):**
```
🧠 MLX Framework: 0.25.2
⚡ Vector Addition: 108,601 vectors/sec
🔍 Query Performance: 112.5 QPS  
💾 Storage Rate: 6,277 vectors/sec
🎯 Speedup: 1.2x (vs. basic implementation)
```

**Geplante Optimierungen (Roadmap):**
```
📈 HNSW Indexing: 5-20x Speedup (logarithmische Suche)
💾 Vector Caching: 3-10x bei wiederholten Queries
⚙️ MLX Compilation: 2-5x durch JIT-Optimierung  
🔄 Batch Processing: 5-15x bei Batch-Operationen
🎯 Ziel: 1000-5000+ QPS
```

---

## 🚀 Schnellstart

### **1. System Requirements**
```bash
# Apple Silicon (M1, M2, M3, etc.)
uname -p  # sollte "arm" ausgeben

# macOS 13.5+ (empfohlen: macOS 14+)
sw_vers

# Python 3.9-3.12 (native ARM64)
python -c "import platform; print(platform.processor())"  # sollte "arm" sein
```

### **2. Installation**
```bash
# Repository klonen
git clone <your-repository-url>
cd mlx-vector-db

# MLX 0.25.2 installieren  
pip install mlx>=0.25.2

# Dependencies installieren
pip install -r requirements.txt

# Konfiguration
cp .env.example .env
# API Key in .env setzen: VECTOR_DB_API_KEY=your-secure-key
```

### **3. Server starten**
```bash
# Server starten
python main.py

# API Dokumentation öffnen
open http://localhost:8000/docs
```

### **4. Erste Schritte**
```python
import requests
import numpy as np

BASE_URL = "http://localhost:8000"
API_KEY = "your-api-key"
headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

# Store erstellen
create_payload = {"user_id": "test_user", "model_id": "test_model"}
response = requests.post(f"{BASE_URL}/admin/create_store", json=create_payload, headers=headers)

# Vektoren hinzufügen (MLX-optimiert)
vectors = np.random.rand(100, 384).astype(np.float32)
metadata = [{"id": f"doc_{i}", "source": "test"} for i in range(100)]

add_payload = {
    "user_id": "test_user",
    "model_id": "test_model", 
    "vectors": vectors.tolist(),
    "metadata": metadata
}
response = requests.post(f"{BASE_URL}/admin/add_test_vectors", json=add_payload, headers=headers)

# Vektoren abfragen
query_vector = vectors[0].tolist()
query_payload = {
    "user_id": "test_user",
    "model_id": "test_model",
    "query": query_vector,
    "k": 5
}
response = requests.post(f"{BASE_URL}/vectors/query", json=query_payload, headers=headers)
results = response.json()
```

---

## 🛠️ API Übersicht

### **Core Endpoints**
- `POST /vectors/add` - Vektoren hinzufügen
- `POST /vectors/query` - Similarity Search
- `POST /vectors/batch_query` - Batch Queries
- `GET /vectors/count` - Store Statistiken

### **Performance Endpoints**
- `GET /performance/health` - MLX System Status  
- `POST /performance/benchmark` - Performance Tests
- `POST /performance/warmup` - MLX Kernel Warmup
- `POST /performance/optimize` - Store Optimierung

### **Admin Endpoints**
- `POST /admin/create_store` - Store Management
- `GET /admin/export_zip` - Backup Export
- `POST /admin/import_zip` - Backup Import  
- `GET /admin/stats` - System Übersicht

### **Monitoring Endpoints**
- `GET /monitoring/health` - Service Health
- `GET /monitoring/metrics` - Prometheus Metriken
- `GET /monitoring/status` - Detaillierter Status

**Vollständige API-Dokumentation:** http://localhost:8000/docs

---

## 📁 Datenstruktur

```
~/.team_mind_data/vector_stores/
├── user_{id}/
│   └── {model_name}/
│       ├── vectors.npz          # MLX-optimierte Vektoren
│       ├── metadata.jsonl       # Zugehörige Metadaten  
│       ├── hnsw_index.pkl       # HNSW Index (geplant)
│       └── .store.lock          # Concurrency Lock
```

**MLX NPZ Format:** Nutzt MLX native serialization für maximale Performance

---

## 🔧 Entwicklung & Optimierung

### **Performance Demo ausführen**
```bash
# Umfassender Performance Test
python performance_demo.py

# Enterprise Features testen  
python enterprise_demo.py

# Basis Funktionalität testen
python demo.py
```

### **Tests ausführen**
```bash
# Unit Tests
pytest tests/

# Spezifische Tests
pytest tests/test_vector_store.py
pytest tests/test_admin_zip_io.py
```

### **Debugging**
```bash
# Debug Routes anzeigen
curl http://localhost:8000/debug/routes

# MLX Status prüfen
python -c "
import mlx.core as mx
print('MLX Version:', mx.__version__)
test = mx.random.normal((10, 384))
mx.eval(test)  
print('✅ MLX working!')
"
```

---

## 🎯 Roadmap

### **Phase 1: Basis (✅ Completed)**
- [x] MLX 0.25.2 Integration
- [x] FastAPI REST API
- [x] Basic Vector Operations
- [x] Authentication System
- [x] Performance Monitoring

### **Phase 2: Performance (🚧 In Progress)**
- [ ] HNSW Indexing Implementation
- [ ] Vector Caching System
- [ ] Advanced MLX Compilation
- [ ] Batch Processing Optimization
- [ ] Memory Management Tuning

### **Phase 3: Enterprise (📋 Planned)**
- [ ] Distributed Storage
- [ ] Advanced Security
- [ ] Custom Index Types
- [ ] Real-time Analytics
- [ ] Auto-scaling

### **Phase 4: AI Integration (🔮 Future)**
- [ ] MLX-LM Integration
- [ ] Custom Model Support
- [ ] Retrieval Optimization
- [ ] Multi-modal Support

---

## 🤝 Contributing

```bash
# Development Setup
git clone <repo>
cd mlx-vector-db
pip install -e ".[dev]"

# Code Style
black .
flake8 .

# Testing
pytest tests/ -v
```

**Beiträge willkommen!** Bitte öffne Issues oder Pull Requests.

---

## 📜 Lizenz

Apache License 2.0 - siehe [LICENSE](LICENSE) Datei.

---

## 🏆 Acknowledgments

* **Apple MLX Team** - Für das fantastische ML Framework
* **FastAPI Community** - Für das moderne Web Framework  
* **NumPy Ecosystem** - Für die Array-Computing Basis

---

## 📞 Support

- **Issues:** GitHub Issues für Bugs und Feature Requests
- **Dokumentation:** http://localhost:8000/docs (wenn Server läuft)
- **Performance:** `python performance_demo.py` für Benchmarks

**🍎 Optimiert für Apple Silicon • 🚀 Powered by MLX 0.25.2**