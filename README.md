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