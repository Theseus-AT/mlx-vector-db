# 🧠 MLXVectorDB für `mlx-langchain-lite`

**MLXVectorDB** ist eine leichtgewichtige, lokal laufende Vektordatenbank zur Verwaltung und Abfrage von Embeddings mit Fokus auf MLX (Apple Silicon & Linux). Sie wurde speziell für den Einsatz in lokalen RAG-Systemen wie `mlx-langchain-lite`, Multi-User-Umgebungen und datenschutzfreundlichen Anwendungen entwickelt.

---

## 🌟 Features

* **MLX-optimiert:** Nutzt `mlx.core` für effiziente Vektoroperationen auf unterstützter Hardware.
* **Lokal & Privat:** Alle Daten bleiben lokal auf deinem System.
* **API-basiert:** Einfache Integration über eine FastAPI-basierte REST-API.
* **Multi-Store:** Unterstützt separate Stores für verschiedene Benutzer und Modelle.
* **Metadaten-Filterung:** Ermöglicht das Filtern von Suchergebnissen basierend auf Metadaten.
* **Import/Export:** Einfaches Sichern und Wiederherstellen von Stores über ZIP-Dateien.
* **Batch & Streaming:** Effiziente Verarbeitung mehrerer Abfragen.

---

## 🚀 Schnellstart

**1. Installation:**

   Klone das Repository und installiere die Abhängigkeiten:

   ```bash
   git clone <dein-repository-url>
   cd mlx-vector-db
   pip install -r requirements.txt

2. Server starten:

Führe das Hauptskript aus, um den FastAPI-Server zu starten:

Bash

python main.py
Der Server läuft standardmäßig unter http://localhost:8000.

3. API-Dokumentation prüfen:

Öffne deinen Browser und gehe zu http://localhost:8000/docs. Dort findest du die interaktive FastAPI-Dokumentation mit allen verfügbaren Endpunkten.

4. Beispiel: Store erstellen und Vektoren hinzufügen (Python requests)

Python

import requests
import numpy as np

BASE_URL = "http://localhost:8000"

user_id = "test_user"
model_id = "test_model"

# Store erstellen (Admin-Endpunkt)
create_payload = {"user_id": user_id, "model_id": model_id}
response = requests.post(f"{BASE_URL}/admin/create_store", json=create_payload)
print("Store erstellen:", response.json())

# Vektoren vorbereiten (Beispiel: 3 Vektoren der Dimension 128)
vectors_np = np.random.rand(3, 128).astype(np.float32)
metadata = [
    {"id": "doc1", "source": "fileA.txt", "chunk": 0},
    {"id": "doc2", "source": "fileA.txt", "chunk": 1},
    {"id": "doc3", "source": "fileB.pdf", "chunk": 0}
]

# Vektoren hinzufügen (Vector-Endpunkt)
add_payload = {
    "user_id": user_id,
    "model_id": model_id,
    "vectors": vectors_np.tolist(), # Als Liste senden
    "metadata": metadata
}
response = requests.post(f"{BASE_URL}/vectors/add", json=add_payload)
print("Vektoren hinzufügen:", response.json())

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
response = requests.post(f"{BASE_URL}/vectors/query", json=query_payload)
print("Abfrageergebnisse:", response.json())
🛠️ API Übersicht
Die API ist in zwei Hauptbereiche unterteilt:

/vectors: Endpunkte für Standard-Vektoroperationen (Hinzufügen, Abfragen, Löschen, Zählen etc.).

/admin: Endpunkte für administrative Aufgaben (Store-Management, Statistiken, Import/Export).

Eine vollständige, interaktive Dokumentation aller Endpunkte, inklusive Schemas und Testmöglichkeiten, findest du unter /docs wenn der Server läuft.

Wichtige Endpunkte:

POST /vectors/add: Fügt Vektoren und Metadaten hinzu.

POST /vectors/query: Sucht nach ähnlichen Vektoren zu einer Abfrage.

POST /vectors/batch_query: Führt mehrere Abfragen gleichzeitig aus.

POST /vectors/stream_query: Führt mehrere Abfragen aus und streamt die Ergebnisse.

POST /vectors/delete: Löscht Vektoren basierend auf Metadaten-Filtern.

GET /vectors/count: Zählt Vektoren/Metadaten in einem Store.

POST /admin/create_store: Erstellt einen neuen, leeren Store.

DELETE /admin/store: Löscht einen gesamten Store.

GET /admin/stats: Gibt aggregierte Statistiken über alle Stores zurück.

GET /admin/export_zip: Exportiert einen Store als ZIP-Datei.

POST /admin/import_zip: Importiert einen Store aus einer ZIP-Datei.

🗂️ Datenstruktur
Die Vektordaten werden lokal gespeichert unter ~/.team_mind_data/vector_stores/. Jeder Store (user_<id>/<modell>) enthält typischerweise:

vectors.npz: Die Vektor-Embeddings als mlx.array in einer NumPy NPZ-Datei.

metadata.jsonl: Die zugehörigen Metadaten, eine Zeile pro Vektor im JSON-Format.

.store.lock: Eine Lock-Datei zur Synchronisierung von Zugriffen.

(Hinweis: Die index.pkl-Datei aus der ursprünglichen README-Struktur scheint in der aktuellen Implementierung nicht aktiv genutzt zu werden)

🤝 Beitrag
Beiträge sind willkommen! Bitte öffne ein Issue oder einen Pull Request.

📜 Lizenz
(Füge hier deine Lizenzinformationen ein, z.B. MIT, Apache 2.0)


**2. API-Dokumentation nutzen**

Das Tolle an FastAPI ist, dass es automatisch eine interaktive API-Dokumentation für dich generiert.

* **Zugriff:** Starte deinen Server (`python main.py`) und öffne `http://localhost:8000/docs` in deinem Webbrowser.
* **Funktionen:**
    * Du siehst alle verfügbaren API-Endpunkte, gruppiert nach Tags (z.B. `admin`, `vector_store`).
    * Für jeden Endpunkt siehst du die HTTP-Methode (GET, POST, etc.), den Pfad, eventuelle Parameter (Pfad, Query, Request Body) und die erwarteten Responses.
    * Du kannst die Request Body Schemas (definiert durch deine Pydantic-Modelle) einsehen.
    * Das Wichtigste: Du kannst die Endpunkte direkt im Browser ausprobieren! Klicke auf "Try it out", gib Beispielwerte ein und klicke auf "Execute". Du siehst dann die Anfrage und die tatsächliche Antwort vom Server.

Du musst also keine separate API-Dokumentation von Hand schreiben. Verweise einfach in deiner `README.md` auf die `/docs`-URL, damit Benutzer sie finden können.

Ich hoffe, das hilft dir, deine Dokumentation zu verbessern! Lass mich wissen, wenn du weitere Fragen hast.
