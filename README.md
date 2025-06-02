# 🧠 MLXVectorDB

**MLXVectorDB** ist eine leichtgewichtige, lokal laufende Vektordatenbank zur Verwaltung und Abfrage von Embeddings mit Fokus auf MLX (Apple Silicon & Linux). Sie wurde speziell für den Einsatz in lokalen RAG-Systemen, Multi-User-Umgebungen und datenschutzfreundlichen Anwendungen entwickelt und ist für hohe Performance optimiert.

---

## 🌟 Features

* **MLX-optimiert:** Nutzt `mlx.core` (Version >= 0.25.2 empfohlen) für effiziente Vektoroperationen auf unterstützter Hardware.
* **Lokal & Privat:** Alle Daten bleiben standardmäßig lokal auf deinem System (`~/.team_mind_data/vector_stores/`).
* **API-basiert:** Einfache Integration über eine FastAPI-basierte REST-API.
* **Multi-Store:** Unterstützt separate Stores für verschiedene Benutzer (`user_id`) und Modelle (`model_id`).
* **Metadaten-Filterung:** Ermöglicht das Filtern von Suchergebnissen basierend auf Metadaten.
* **Import/Export:** Einfaches Sichern und Wiederherstellen von Stores über ZIP-Dateien (Admin-Funktion).
* **Batch & Streaming:** Effiziente Verarbeitung mehrerer Abfragen.
* **Performance-Optimierungen:**
    * **HNSW Indexing:** Schnelle Ähnlichkeitssuche durch Hierarchical Navigable Small World Graphen.
    * **Vector Caching:** Intelligentes Caching von Vektoren und Metadaten zur Reduzierung von Disk I/O.
    * **Kompilierte MLX Operationen:** Nutzung von `@mx.compile` für beschleunigte Berechnungen.
* **Monitoring & Health Checks:** Endpunkte zur Überwachung von Systemgesundheit und Metriken.
* **Konfigurierbar:** Erweiterte Einstellungen über Umgebungsvariablen und Konfigurationsdateien.

---

## 🚀 Schnellstart

**1. Installation:**

Klone das Repository und installiere die Abhängigkeiten:

```bash
git clone <dein-repository-url>
cd mlx-vector-db
pip install -r requirements.txt

Stelle sicher, dass du mlx>=0.25.2 verwendest, wie in der requirements.txt spezifiziert.

2. Server starten:

Führe das Hauptskript aus, um den FastAPI-Server zu starten:

Bash

python main.py
Der Server läuft standardmäßig unter http://localhost:8000.

3. API-Dokumentation prüfen:

Öffne deinen Browser und gehe zu http://localhost:8000/docs. Dort findest du die interaktive FastAPI-Dokumentation mit allen verfügbaren Endpunkten und deren Schemas.

4. Beispiel: Store erstellen und Vektoren hinzufügen (Python requests)

Python

import requests
import numpy as np

BASE_URL = "http://localhost:8000"
# Definiere deinen API-Key (aus .env oder Konfiguration)
# Für Admin-Endpunkte wird ein API-Key benötigt.
# Für dieses Beispiel nehmen wir an, dass die Authentifizierung konfiguriert ist.
# headers = {"X-API-Key": "DEIN_API_KEY"} # oder "Authorization": "Bearer DEIN_API_KEY"

user_id = "demo_user"
model_id = "demo_model"

# Store erstellen (Admin-Endpunkt)
# Stelle sicher, dass der Admin-Endpunkt korrekt gesichert ist und du ggf. Header verwendest.
# Für dieses Beispiel lassen wir die Header weg, gehe davon aus, dass Auth deaktiviert ist oder der Key anders bereitgestellt wird.
create_payload = {"user_id": user_id, "model_id": model_id}
# Admin-Endpunkte benötigen einen API-Key.
# Ersetze "YOUR_ADMIN_API_KEY" mit deinem tatsächlichen Admin-API-Key.
# Wenn du die Sicherheit in settings.py deaktiviert hast (enable_auth=False),
# werden die Header nicht benötigt.
admin_headers = {"X-API-Key": "YOUR_ADMIN_API_KEY"}


response = requests.post(f"{BASE_URL}/admin/create_store", json=create_payload, headers=admin_headers)
if response.status_code == 200:
    print("Store erstellen:", response.json())
elif response.status_code == 401:
    print("Store erstellen: Authentifizierung fehlgeschlagen. Stelle sicher, dass dein API Key korrekt ist.")
    exit() # Beende das Skript, wenn die Authentifizierung fehlschlägt
elif response.status_code == 409: # Store existiert bereits
    print("Store erstellen: Store existiert bereits oder konnte nicht erstellt werden.", response.json())
    # Fahre fort, da der Store für das Demo eventuell schon existiert
else:
    print("Store erstellen: Fehler", response.status_code, response.text)
    exit()


# Vektoren vorbereiten (Beispiel: 3 Vektoren der Dimension 128)
vectors_np = np.random.rand(3, 128).astype(np.float32)
metadata = [
    {"id": "doc1", "source": "fileA.txt", "chunk": 0},
    {"id": "doc2", "source": "fileA.txt", "chunk": 1},
    {"id": "doc3", "source": "fileB.pdf", "chunk": 0}
]

# Vektoren hinzufügen (Vector-Endpunkt)
# Dieser Endpunkt könnte je nach Konfiguration auch einen API-Key benötigen.
# Passe die Header entsprechend an, falls nötig.
# Für dieses Beispiel gehen wir davon aus, dass /vectors/add nicht explizit Admin-geschützt ist
# oder der Key global für alle gesicherten Endpunkte gilt.
add_payload = {
    "user_id": user_id,
    "model_id": model_id,
    "vectors": vectors_np.tolist(), # Als Liste senden
    "metadata": metadata
}
# Verwende hier ggf. die gleichen Header wie für den Admin-Endpunkt oder spezifische Header für Vektor-Endpunkte
response = requests.post(f"{BASE_URL}/vectors/add", json=add_payload, headers=admin_headers) # Annahme: admin_headers gelten auch hier oder anpassen
if response.status_code == 200:
    print("Vektoren hinzufügen:", response.json())
else:
    print("Vektoren hinzufügen: Fehler", response.status_code, response.text)
    exit()

# Vektoren abfragen (Vector-Endpunkt)
query_vector = vectors_np[0].tolist() # Einzelner Vektor als Liste
query_payload = {
    "user_id": user_id,
    "model_id": model_id,
    "query": query_vector,
    "k": 2,
    # Optional: Metadaten-Filter
    # "filter_metadata": {"source": "fileA.txt"}
}
response = requests.post(f"{BASE_URL}/vectors/query", json=query_payload, headers=admin_headers) # Annahme: admin_headers gelten auch hier oder anpassen
if response.status_code == 200:
    print("Abfrageergebnisse:", response.json())
else:
    print("Abfrageergebnisse: Fehler", response.status_code, response.text)

Hinweis zum API-Key: Viele Endpunkte, insbesondere unter /admin, sind durch einen API-Key geschützt. Stelle sicher, dass der VECTOR_DB_API_KEY in deiner Umgebung oder .env-Datei gesetzt ist und du ihn in den Headern deiner Anfragen sendest (X-API-Key oder Authorization: Bearer <key>).

🛠️ API Übersicht
Die API ist in mehrere Bereiche unterteilt, die über verschiedene Router bereitgestellt werden. Eine vollständige, interaktive Dokumentation aller Endpunkte, inklusive Schemas und Testmöglichkeiten, findest du unter /docs, wenn der Server läuft.

/vectors - Standard Vektoroperationen
POST /vectors/add: Fügt Vektoren und Metadaten hinzu.
POST /vectors/query: Sucht nach ähnlichen Vektoren zu einer Abfrage.
POST /vectors/batch_query: Führt mehrere Abfragen gleichzeitig aus.
POST /vectors/stream_query: Führt mehrere Abfragen aus und streamt die Ergebnisse.
POST /vectors/delete: Löscht Vektoren basierend auf Metadaten-Filtern.
POST /vectors/create: Erstellt einen neuen Store (benutzerseitig).
GET /vectors/count: Zählt Vektoren/Metadaten in einem Store.
GET /vectors/users: Listet alle Benutzer auf.
GET /vectors/models: Listet Modelle für einen bestimmten Benutzer auf.
GET /vectors/ping: Einfacher Ping-Check für diesen Router.
/admin - Administrative Aufgaben
POST /admin/create_store: Erstellt einen neuen, leeren Store.
DELETE /admin/store: Löscht einen gesamten Store.
GET /admin/stats: Gibt aggregierte Statistiken über alle Stores zurück.
GET /admin/store/stats: Gibt Statistiken für einen spezifischen Store zurück.
POST /admin/add_test_vectors: Fügt Test-Vektoren hinzu (für Debugging/Tests).
GET /admin/export_zip: Exportiert einen Store als ZIP-Datei.
POST /admin/import_zip: Importiert einen Store aus einer ZIP-Datei.
GET /admin/health: Health-Check für den Admin-Service.
/performance - Performance-Monitoring und Optimierung
GET /performance/stats: Umfassende Performance-Statistiken (MLX-Systeminfo etc.).
GET /performance/cache/stats: Detaillierte Cache-Statistiken.
POST /performance/cache/clear: Leert den globalen Cache.
POST /performance/warmup: Wärmt kompilierte MLX-Funktionen auf.
POST /performance/benchmark: Führt Performance-Benchmarks durch.
POST /performance/optimize: Optimiert einen spezifischen Vektor-Store (z.B. HNSW-Index-Neubau).
GET /performance/health: Health-Check für das Performance-Subsystem.
/monitoring - Metriken und Systemüberwachung
GET /monitoring/health: Basis-Health-Check (ohne Authentifizierung).
GET /monitoring/health/detailed: Detaillierter Health-Check mit Komponentenstatus (Authentifizierung erforderlich).
GET /monitoring/metrics: Liefert Anwendungsmetriken im JSON- oder Prometheus-Format.
GET /monitoring/metrics/summary: Zusammenfassung wichtiger Metriken.
GET /monitoring/status: Umfassender Service-Status inklusive Performance-Indikatoren.
POST /monitoring/alerts/test: Testet das Alert-System.
Allgemeine Endpunkte
GET /: Root-Endpunkt mit Basisinformationen.
GET /health: Einfacher Health-Check für die gesamte Anwendung.
GET /debug/routes: Zeigt alle registrierten Routen (nützlich für Debugging).
🗂️ Datenstruktur
Die Vektordaten werden lokal im Verzeichnis ~/.team_mind_data/vector_stores/ gespeichert (konfigurierbar über VECTOR_STORE_BASE_PATH in config/settings.py oder Umgebungsvariablen). Jeder Store (user_<id>/<modell>/) enthält:

vectors.npz: Die Vektor-Embeddings als mlx.array, gespeichert in einer NumPy NPZ-Datei.
metadata.jsonl: Die zugehörigen Metadaten, eine Zeile pro Vektor im JSON-Format.
hnsw_index.pkl: Die serialisierte Datei für den HNSW-Index, falls dieser für den Store erstellt wurde.
.store.lock: Eine Lock-Datei zur Synchronisierung von Zugriffen auf den Store.
⚡ Performance Features
MLXVectorDB integriert mehrere Mechanismen zur Leistungssteigerung:

HNSW Indexing (hnsw_index.py): Ein Hierarchical Navigable Small World (HNSW) Graph wird für eine schnelle Ähnlichkeitssuche (ANN) verwendet. Dies reduziert die Suchzeit von linearer auf logarithmische Komplexität. Der Index wird automatisch für größere Stores aufgebaut und kann über den /performance/optimize-Endpunkt verwaltet werden.
Vector Cache (vector_cache.py): Ein LRU-Cache (Least Recently Used) hält häufig abgerufene Vektoren und Metadaten im Speicher. Dies minimiert teure Festplattenzugriffe. Der Cache ist global und seine Größe ist konfigurierbar.
MLX Compiled Operations (mlx_optimized.py): Kernoperationen wie Kosinus-Ähnlichkeitsberechnung und Top-K-Selektion sind mit @mx.compile für die Apple Silicon Architektur optimiert.
Optimized Vector Store (optimized_vector_store.py): Eine leistungsoptimierte Implementierung der Vektor-Speicherlogik, die HNSW, Caching und kompilierte Funktionen kombiniert.
⚙️ Konfiguration
Die Anwendung kann über Umgebungsvariablen oder eine zentrale Konfigurationsdatei (config/settings.py) angepasst werden. Wichtige Einstellungen umfassen:

Server-Einstellungen (HOST, PORT)
Sicherheit (VECTOR_DB_API_KEY, ENABLE_AUTH)
Performance-Parameter (Cache-Größe, HNSW-Einstellungen)
Speicherpfade (VECTOR_STORE_BASE_PATH)
Monitoring-Optionen
Siehe config/settings.py für eine vollständige Liste und Standardwerte.

📊 Monitoring
Die Datenbank bietet Endpunkte unter /monitoring für detaillierte Einblicke in den Systemstatus, Performance-Metriken (auch im Prometheus-Format) und Health-Checks. Diese sind nützlich für Produktionseinsätze und Debugging.

💡 Demos
Das Repository enthält verschiedene Demo-Skripte, um die Funktionalität zu veranschaulichen:

demo.py: Zeigt grundlegende Operationen wie das Erstellen von Stores, Hinzufügen und Abfragen von Vektoren.
performance_demo.py: Demonstriert die Performance-Features und führt Benchmarks aus.
enterprise_demo.py: Zeigt erweiterte Funktionen wie Monitoring, Metriken und Health-Checks.
🤝 Beitrag
Beiträge sind willkommen! Bitte öffne ein Issue oder einen Pull Request.

