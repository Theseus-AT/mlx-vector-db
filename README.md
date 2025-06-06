# 🧠 MLX Vector Database für Apple Silicon

**High-Performance Vector Database optimiert für Apple Silicon mit MLX 0.25.2**

MLXVectorDB ist eine native Apple Silicon Vector Database, die das MLX Machine Learning Framework nutzt für maximale Performance auf M-Series Chips. Entwickelt für lokale RAG-Systeme, Multi-User-Umgebungen und datenschutzfreundliche AI-Anwendungen.

## 🚀 Schnellstart

### **Voraussetzungen**
```bash
# Apple Silicon (M1, M2, M3, etc.)
uname -p  # sollte "arm" ausgeben

# macOS 13.5+ (empfohlen: macOS 14+)
sw_vers

# Python 3.9-3.12 (native ARM64)
python -c "import platform; print(platform.processor())"  # sollte "arm" sein
```

### **Installation**
```bash
# Repository klonen
git clone <your-repository-url>
cd mlx-vector-db

# MLX installieren
pip install mlx>=0.25.2

# Core Dependencies installieren
pip install -r requirements.txt

# Konfiguration
cp .env.example .env
# API Key in .env setzen: VECTOR_DB_API_KEY=your-secure-key
```

### **Server starten**
```bash
# Server starten
python main.py

# API Dokumentation öffnen
open http://localhost:8000/docs
```

### **Ersten Test ausführen**
```bash
# Einfacher Funktionstest
python simple_test.py

# Vollständiger Test
python working_test_fixed.py

# Sprint 3 Demo (mit allen Features)
python sprint3_demo.py
```

---

## 🌟 Features

### **🚀 Apple Silicon Optimiert**
* **MLX 0.25.2**: Native Apple Machine Learning Framework
* **Unified Memory**: Zero-copy Operationen zwischen CPU/GPU
* **Metal Kernels**: GPU-Beschleunigung mit Kernel-Caching
* **Lazy Evaluation**: Arrays werden nur bei Bedarf materialisiert
* **ARM64 Native**: Maximale Hardware-Ausnutzung

### **⚡ High-Performance**
* **1000+ vectors/sec**: Vector Addition Rate
* **100+ QPS**: Query Performance (optimierbar auf 1000+ QPS)
* **Sub-10ms Latency**: Einzelne Queries
* **Batch Processing**: Effiziente Multi-Query Verarbeitung
* **MLX Compilation**: JIT-optimierte Operationen

### **🏗️ Production-Ready**
* **FastAPI REST API**: Async, OpenAPI dokumentiert
* **Rate Limiting**: Intelligente Request-Kontrolle
* **Multi-Store Support**: Separate Stores für User/Modelle
* **Authentication**: API-Key basierte Sicherheit
* **Error Handling**: Graceful Degradation
* **Python SDK**: Async Client mit One-liner Methods

### **🔒 Privatsphäre & Kontrolle**
* **100% Lokal**: Alle Daten bleiben auf dem System
* **Keine Cloud-Abhängigkeit**: Komplett offline betreibbar
* **Metadaten-Filterung**: Präzise Suchkriterien
* **Multi-Tenant**: Sichere User-Isolation

---

## 📊 Performance Benchmarks

**Aktuelle Performance (Korrigierte Version):**
```
🧠 MLX Framework: 0.25.2
⚡ Vector Addition: 1000+ vectors/sec
🔍 Query Performance: 100+ QPS  
💾 Latency: <10ms average
🎯 Success Rate: >99%
```

---

## 🛠️ API Übersicht

### **Core Endpoints**
- `POST /vectors/add` - Vektoren hinzufügen
- `POST /vectors/query` - Similarity Search
- `POST /vectors/batch_query` - Batch Queries
- `GET /vectors/count` - Store Statistiken

### **Admin Endpoints**
- `POST /admin/create_store` - Store Management
- `DELETE /admin/store` - Store löschen
- `GET /admin/store/stats` - Store Statistiken
- `GET /admin/list_stores` - Alle Stores auflisten

### **Monitoring Endpoints**
- `GET /health` - Service Health
- `GET /performance/health` - Performance Status
- `GET /monitoring/metrics` - System Metriken
- `GET /system/info` - System Information

**Vollständige API-Dokumentation:** http://localhost:8000/docs

---

## 🔧 Erste Schritte

### **1. Einfacher Test**
```bash
python simple_test.py
```
Führt grundlegende Funktionalitätstests durch und überprüft, ob alles korrekt funktioniert.

### **2. Store erstellen und verwenden**
```python
import requests
import numpy as np

BASE_URL = "http://localhost:8000"
API_KEY = "mlx-vector-dev-key-2024"
headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

# Store erstellen
create_payload = {"user_id": "my_user", "model_id": "my_model", "dimension": 384}
response = requests.post(f"{BASE_URL}/admin/create_store", json=create_payload, headers=headers)

# Vektoren hinzufügen
vectors = np.random.rand(100, 384).astype(np.float32)
metadata = [{"id": f"doc_{i}", "text": f"Document {i}"} for i in range(100)]

add_payload = {
    "user_id": "my_user",
    "model_id": "my_model", 
    "vectors": vectors.tolist(),
    "metadata": metadata
}
response = requests.post(f"{BASE_URL}/vectors/add", json=add_payload, headers=headers)

# Vektoren abfragen
query_vector = vectors[0].tolist()
query_payload = {
    "user_id": "my_user",
    "model_id": "my_model",
    "query": query_vector,
    "k": 5
}
response = requests.post(f"{BASE_URL}/vectors/query", json=query_payload, headers=headers)
results = response.json()
```

### **3. Python SDK verwenden**
```python
import asyncio
from sdk.python.mlx_vector_client import create_client

async def main():
    async with create_client("http://localhost:8000", api_key="mlx-vector-dev-key-2024") as client:
        # Store erstellen
        await client.create_store("my_user", "my_model", dimension=384)
        
        # Einfache Text-Embeddings hinzufügen
        texts = ["Machine learning is amazing", "Vector search is powerful"]
        embeddings = [[0.1] * 384, [0.2] * 384]  # Ihre echten Embeddings hier
        
        await client.quick_add("my_user", "my_model", texts, embeddings)
        
        # Semantic Search
        results = await client.quick_search("my_user", "my_model", embeddings[0], k=5)
        print("Search results:", results)

asyncio.run(main())
```

---

## 📁 Datenstruktur

```
~/.team_mind_data/vector_stores/
├── user_{id}/
│   └── {model_name}/
│       ├── vectors.npz          # MLX-optimierte Vektoren
│       └── metadata.jsonl       # Zugehörige Metadaten  
```

**MLX NPZ Format:** Nutzt MLX native serialization für maximale Performance

---

## 🔧 Entwicklung & Testing

### **Tests ausführen**
```bash
# Einfacher Funktionstest
python simple_test.py

# Vollständiger API-Test
python working_test_fixed.py

# Sprint 3 Demo (alle Features)
python sprint3_demo.py

# Unit Tests (falls verfügbar)
pytest tests/ -v
```

### **Performance Demo**
```bash
# Basis Demo
python demo.py

# Performance-spezifische Tests
python performance_demo.py
```

### **Debugging**
```bash
# MLX Status prüfen
python -c "
import mlx.core as mx
print('MLX Version:', mx.__version__)
test = mx.random.normal((10, 384))
mx.eval(test)  
print('✅ MLX working!')
"

# Server Debug-Info (nur Development)
curl http://localhost:8000/debug/mlx
curl http://localhost:8000/debug/routes
```

---

## 🚨 Troubleshooting

### **Häufige Probleme**

#### **MLX Import Fehler**
```bash
# Lösung: MLX neu installieren
pip uninstall mlx
pip install mlx>=0.25.2

# Verify Installation
python -c "import mlx.core as mx; print(mx.default_device())"
```

#### **Server startet nicht**
```bash
# Ports prüfen
lsof -i :8000

# Dependencies prüfen
pip install -r requirements.txt

# Logs prüfen
python main.py 2>&1 | tee server.log
```

#### **API Authentifizierung Fehler**
```bash
# .env Datei prüfen
cat .env

# API Key in Environment setzen
export VECTOR_DB_API_KEY=mlx-vector-dev-key-2024

# Test mit curl
curl -H "Authorization: Bearer mlx-vector-dev-key-2024" http://localhost:8000/health
```

#### **Performance Probleme**
```bash
# System Resources prüfen
python -c "
import psutil
print('Memory:', psutil.virtual_memory().percent, '%')
print('CPU:', psutil.cpu_percent(), '%')
"

# MLX Device prüfen
python -c "
import mlx.core as mx
print('Device:', mx.default_device())
"
```

### **Support**
- **Issues:** GitHub Issues für Bugs und Feature Requests
- **Tests:** `python simple_test.py` für grundlegende Funktionalität
- **Performance:** `python working_test_fixed.py` für vollständige Tests
- **Logs:** Server Logs unter `/logs/` (wenn konfiguriert)

---

## 🎯 Roadmap

### **Phase 1: Core Stability (✅ Completed)**
- [x] MLX 0.25.2 Integration
- [x] FastAPI REST API
- [x] Basic Vector Operations
- [x] Authentication System
- [x] Error Handling & Recovery

### **Phase 2: Production Features (✅ Completed)**
- [x] Rate Limiting
- [x] Python SDK
- [x] Batch Operations
- [x] Performance Monitoring
- [x] Health Checks

### **Phase 3: Advanced Features (🚧 In Progress)**
- [x] MLX-LM Integration (Basic)
- [x] RAG Pipeline (Simplified)
- [x] Text Processing (Mock/Fallback)
- [ ] HNSW Indexing (Planned)
- [ ] Advanced Caching (Planned)

### **Phase 4: Enterprise (📋 Planned)**
- [ ] Distributed Storage
- [ ] Advanced Security
- [ ] Real-time Analytics
- [ ] Auto-scaling

---

## 📦 Dependencies

### **Core Requirements**
```
mlx>=0.25.2
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0
numpy>=1.26.0
psutil>=5.9.0
```

### **Optional Dependencies**
```bash
# Für MLX-LM Integration
pip install mlx-lm sentence-transformers

# Für erweiterte Features
pip install matplotlib tqdm

# Für Tests
pip install pytest pytest-asyncio httpx
```

---

## 🏗️ Architektur

```
MLX Vector Database
├── main.py                     # FastAPI Application
├── api/
│   ├── routes/
│   │   ├── vectors.py         # Core Vector Operations
│   │   ├── admin.py           # Store Management
│   │   ├── performance.py     # Performance Endpoints
│   │   └── monitoring.py      # Health & Monitoring
│   └── middleware/
│       └── rate_limiting.py   # Rate Limiting Logic
├── service/
│   ├── optimized_vector_store.py  # MLX Vector Store
│   └── models.py              # Pydantic Models
├── sdk/python/
│   └── mlx_vector_client.py   # Python SDK
├── integrations/
│   └── mlx_lm_pipeline.py     # MLX-LM Integration
├── security/
│   └── auth.py                # Authentication
└── tests/
    ├── simple_test.py         # Basic Functionality
    ├── working_test_fixed.py  # Complete API Tests
    └── sprint3_demo.py        # Feature Demo
```

---

## 🔐 Sicherheit

### **Production Deployment**
```bash
# Environment Variables setzen
export VECTOR_DB_API_KEY=your-secure-api-key
export VECTOR_DB_ADMIN_KEY=your-admin-key
export ENVIRONMENT=production

# Production Server starten
python main.py production
```

### **API Keys**
- **Development:** `mlx-vector-dev-key-2024` (Standard)
- **Production:** Eigene sichere Keys verwenden
- **Admin Operations:** Separate Admin-Keys für kritische Operationen

### **Best Practices**
- API Keys niemals in Code committen
- `.env` Dateien zu `.gitignore` hinzufügen
- Rate Limiting in Production aktivieren
- Health Checks für Monitoring einrichten

---

## 🚀 Produktions-Deployment

### **Docker Deployment**
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "main.py", "production"]
```

### **Systemd Service**
```ini
[Unit]
Description=MLX Vector Database
After=network.target

[Service]
Type=simple
User=mlxvector
WorkingDirectory=/opt/mlx-vector-db
Environment=ENVIRONMENT=production
ExecStart=/opt/mlx-vector-db/venv/bin/python main.py production
Restart=always

[Install]
WantedBy=multi-user.target
```

---

## 📈 Performance Tuning

### **MLX Optimierungen**
```python
# In optimized_vector_store.py
config = MLXVectorStoreConfig(
    jit_compile=True,          # JIT Compilation aktivieren
    use_metal=True,            # Metal GPU nutzen
    enable_lazy_eval=True,     # Lazy Evaluation
    max_cache_vectors=10000,   # Vector Caching
    batch_size=1000           # Optimale Batch-Größe
)
```

### **System Tuning**
```bash
# Memory für MLX optimieren
export MLX_MEMORY_POOL_SIZE=2GB

# Metal Performance Shaders
export METAL_DEVICE_WRAPPER_TYPE=1

# System Limits erhöhen
ulimit -n 65536
```

---

## 📚 Weitere Ressourcen

### **MLX Framework**
- [MLX GitHub](https://github.com/ml-explore/mlx)
- [MLX-LM Examples](https://github.com/ml-explore/mlx-lm)
- [MLX Documentation](https://ml-explore.github.io/mlx/build/html/)

### **Beispiel-Integrationen**
- **Embedding Models:** sentence-transformers, E5, BGE
- **Vector Search:** Semantic Search, RAG Pipelines
- **Text Processing:** Document Chunking, Metadata Extraction

### **Community**
- **Issues:** Bug Reports und Feature Requests via GitHub
- **Discussions:** Für Fragen und Diskussionen
- **Contributions:** Pull Requests willkommen!

---

## 📜 Lizenz

Apache License 2.0 - siehe [LICENSE](LICENSE) Datei.

---

## 🏆 Acknowledgments

* **Apple MLX Team** - Für das fantastische ML Framework
* **FastAPI Community** - Für das moderne Web Framework  
* **NumPy Ecosystem** - Für die Array-Computing Basis

---

**🍎 Optimiert für Apple Silicon • 🚀 Powered by MLX 0.25.2**

## 🎯 Nächste Schritte

1. **Erste Tests:** `python simple_test.py`
2. **Vollständige Demo:** `python sprint3_demo.py`  
3. **Eigene Integration:** SDK verwenden oder REST API direkt nutzen
4. **Production:** Environment Variables setzen und optimieren
5. **Erweiterte Features:** MLX-LM Dependencies installieren für vollständige Text-Pipeline