# Neue Datei: run_demos_cli.py
# Ein zentraler Runner für die verschiedenen Demo-Skripte.

import argparse
import sys
import logging
from demo import main as run_full_basic_demo
from performance_demo_api import run_performance_demo as run_api_performance_demo
from enterprise_demo import main as run_enterprise_demo
from performance_demo_hnsw import main as run_hnsw_performance_tests

# Konfiguriere ein einfaches Logging für das CLI-Tool selbst
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DemoRunner")

# Importiere die Hauptfunktionen aus Ihren Demo-Dateien
# Stellen Sie sicher, dass diese Dateien im Python-Pfad sind oder passen Sie die Importe an.
try:
    from demo import main as run_full_basic_demo # Annahme: demo.py hat eine main() Funktion
    # oder spezifische Funktionen: from demo import run_basic_demo, run_performance_demo as run_basic_perf_demo, run_advanced_demo
    DEMO_PY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Konnte 'demo.py' nicht importieren: {e}. Die Basis-Demos sind nicht verfügbar.")
    DEMO_PY_AVAILABLE = False

try:
    # Unterscheidung: dies ist die API-basierte Performance-Demo
    from performance_demo_api import run_performance_demo as run_api_performance_demo
    # Benennen Sie Ihre API-basierte performance_demo.py ggf. um zu performance_demo_api.py,
    # um Konflikte mit der HNSW-spezifischen performance_demo.py zu vermeiden.
    PERFORMANCE_DEMO_API_PY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Konnte 'performance_demo_api.py' (API-Version) nicht importieren: {e}. Die API-Performance-Demo ist nicht verfügbar.")
    PERFORMANCE_DEMO_API_PY_AVAILABLE = False

try:
    from enterprise_demo import main as run_enterprise_demo
    ENTERPRISE_DEMO_PY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Konnte 'enterprise_demo.py' nicht importieren: {e}. Die Enterprise-Demo ist nicht verfügbar.")
    ENTERPRISE_DEMO_PY_AVAILABLE = False

try:
    # Dies ist die HNSW-spezifische Performance-Demo, die Sie zuletzt hochgeladen haben.
    from performance_demo_hnsw import main as run_hnsw_performance_tests
    # Benennen Sie Ihre HNSW-spezifische performance_demo.py zu performance_demo_hnsw.py um.
    PERFORMANCE_DEMO_HNSW_PY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Konnte 'performance_demo_hnsw.py' (HNSW-Version) nicht importieren: {e}. Die HNSW-Performance-Tests sind nicht verfügbar.")
    PERFORMANCE_DEMO_HNSW_PY_AVAILABLE = False


def main():
    parser = argparse.ArgumentParser(
        description="MLX Vector DB Demo Runner. Führt verschiedene Demo- und Test-Skripte aus.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Argumente für die Auswahl der Demos
    # Die "Checkliste" wird durch die Hilfeausgabe dieses Parsers realisiert.
    parser.add_argument(
        "--demo",
        choices=["basic", "api_perf", "enterprise", "hnsw_perf", "all_api"],
        help=(
            "Wählt die auszuführende Demo aus:\n"
            "  basic       - Führt die grundlegenden Store-Operationen aus (aus demo.py).\n"
            "  api_perf    - Führt die API-Performance-Benchmarks aus (aus performance_demo_api.py).\n"
            "  enterprise  - Führt die Enterprise-Features-Demo aus (aus enterprise_demo.py).\n"
            "  hnsw_perf   - Führt die HNSW-spezifischen Performance-Tests aus (aus performance_demo_hnsw.py).\n"
            "  all_api     - Führt alle API-basierten Demos aus (basic, api_perf, enterprise).\n"
        )
    )
    # Hier könnten weitere Argumente hinzugefügt werden, z.B. um Parameter an die Demos zu übergeben.

    args = parser.parse_args()

    if not any([DEMO_PY_AVAILABLE, PERFORMANCE_DEMO_API_PY_AVAILABLE, ENTERPRISE_DEMO_PY_AVAILABLE, PERFORMANCE_DEMO_HNSW_PY_AVAILABLE]):
        logger.error("Keine Demo-Module gefunden. Bitte stellen Sie sicher, dass die Demo-Dateien korrekt importiert werden können.")
        sys.exit(1)

    if not args.demo:
        logger.info("Keine spezifische Demo ausgewählt. Zeige Hilfe:")
        parser.print_help()
        sys.exit(0)

    logger.info(f"Ausgewählte Demo: {args.demo}")

    if args.demo == "basic" or args.demo == "all_api":
        if DEMO_PY_AVAILABLE:
            logger.info("\n--- Starte Basis-Demo (aus demo.py) ---")
            run_full_basic_demo()
            logger.info("--- Basis-Demo beendet ---\n")
        else:
            logger.warning("Basis-Demo (demo.py) nicht verfügbar.")
    
    if args.demo == "api_perf" or args.demo == "all_api":
        if PERFORMANCE_DEMO_API_PY_AVAILABLE:
            logger.info("\n--- Starte API Performance Demo (aus performance_demo_api.py) ---")
            # Ihre API-basierte performance_demo.py heißt hier performance_demo_api.py
            # und hat idealerweise eine run_performance_demo() Funktion
            run_api_performance_demo() # Passen Sie den Funktionsnamen ggf. an
            logger.info("--- API Performance Demo beendet ---\n")
        else:
            logger.warning("API Performance Demo (performance_demo_api.py) nicht verfügbar.")

    if args.demo == "enterprise" or args.demo == "all_api":
        if ENTERPRISE_DEMO_PY_AVAILABLE:
            logger.info("\n--- Starte Enterprise Demo (aus enterprise_demo.py) ---")
            run_enterprise_demo()
            logger.info("--- Enterprise Demo beendet ---\n")
        else:
            logger.warning("Enterprise Demo (enterprise_demo.py) nicht verfügbar.")

    if args.demo == "hnsw_perf":
        if PERFORMANCE_DEMO_HNSW_PY_AVAILABLE:
            logger.info("\n--- Starte HNSW Performance Tests (aus performance_demo_hnsw.py) ---")
            # Ihre HNSW-spezifische performance_demo.py heißt hier performance_demo_hnsw.py
            run_hnsw_performance_tests()
            logger.info("--- HNSW Performance Tests beendet ---\n")
        else:
            logger.warning("HNSW Performance Tests (performance_demo_hnsw.py) nicht verfügbar.")
            
    logger.info("Demo-Ausführung beendet.")

if __name__ == "__main__":
    main()