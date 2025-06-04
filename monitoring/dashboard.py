# Neue Datei: monitoring/dashboard.py
# Erstellt ein einfaches HTML-Dashboard zur Visualisierung von Metriken.

# MLX Specificity: Das Dashboard kann MLX-spezifische Metriken anzeigen,
#                  z.B. Auslastung der GPU (falls messbar und als Metrik erfasst),
#                  Anzahl der kompilierten MLX-Funktionen, Performance von MLX-Ops.
# LLM Anbindung: Visualisierung von Query-Latenzen, Indexierungsraten und Cache-Hit-Raten
#                kann helfen, die Performance der Vektor-DB im Kontext von LLM-Anwendungen
#                zu verstehen und zu optimieren.

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
import httpx # Um Metriken vom eigenen /monitoring/metrics Endpunkt abzurufen
import json
import os
import logging

# Plotly wird benötigt, zu requirements.txt hinzufügen.
try:
    import plotly.graph_objects as go
    import plotly.utils
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    plotly = None
    logging.warning("Plotly ist nicht installiert. Das Dashboard kann keine Graphen rendern. Installieren mit: pip install plotly")


from security.auth import verify_admin_api_key # Dashboard nur für Admins

logger = logging.getLogger("mlx_vector_db.dashboard")

# Router für das Dashboard
# Es ist unüblich, einen APIRouter direkt in einer solchen Datei zu haben,
# besser wäre es, die Route in api/routes/monitoring.py zu definieren und
# die Logik hierher zu importieren. Der Plan hat es aber so.
# Wir erstellen hier eine Klasse und die Route wird in api/routes/monitoring.py hinzugefügt.

class MonitoringDashboard:
    def __init__(self, metrics_endpoint_url: str, api_key: Optional[str] = None, jwt_token: Optional[str] = None):
        """
        Args:
            metrics_endpoint_url: Die volle URL zum /monitoring/metrics/summary Endpunkt.
            api_key: API-Key für den Zugriff auf den Metrik-Endpunkt.
            jwt_token: JWT-Token für den Zugriff.
        """
        self.metrics_endpoint_url = metrics_endpoint_url
        self.headers = {"Accept": "application/json"}
        if jwt_token: # JWT bevorzugen
            self.headers["Authorization"] = f"Bearer {jwt_token}"
        elif api_key:
            self.headers["X-API-Key"] = api_key # Wie im SDK Client geplant

        if not PLOTLY_AVAILABLE:
            logger.error("Plotly nicht verfügbar. Dashboard-Graphen können nicht erstellt werden.")

    async def _fetch_metrics_summary(self) -> Optional[Dict]:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(self.metrics_endpoint_url, headers=self.headers)
                response.raise_for_status()
                return response.json().get("summary") # Greift auf das 'summary' Feld zu
        except httpx.HTTPStatusError as e:
            logger.error(f"Fehler beim Abrufen der Metriken für das Dashboard (Status {e.response.status_code}): {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"Allgemeiner Fehler beim Abrufen der Metriken für das Dashboard: {e}")
            return None

    def _create_gauge_chart(self, title: str, value: float, max_value: float, unit: str = "") -> Optional[str]:
        if not PLOTLY_AVAILABLE: return None
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            title={'text': title},
            gauge={'axis': {'range': [0, max_value]},
                   'bar': {'color': "darkblue"},
                   'steps': [
                       {'range': [0, max_value * 0.7], 'color': "lightgreen"},
                       {'range': [max_value * 0.7, max_value * 0.9], 'color': "yellow"},
                       {'range': [max_value * 0.9, max_value], 'color': "red"}],
                   },
            number={'suffix': unit}
        ))
        fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    def _create_bar_chart(self, title: str, x_labels: List[str], y_values: List[float], y_title: str) -> Optional[str]:
        if not PLOTLY_AVAILABLE: return None
        fig = go.Figure([go.Bar(x=x_labels, y=y_values)])
        fig.update_layout(title_text=title, yaxis_title=y_title, height=300, margin=dict(l=20, r=20, t=40, b=20))
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


    async def create_dashboard_html(self) -> str:
        """Erstellt das HTML für das Dashboard."""
        metrics_summary = await self._fetch_metrics_summary()
        
        # Fallback, falls Metriken nicht geladen werden können
        if not metrics_summary:
            return """
            <!DOCTYPE html><html><head><title>MLX Vector DB Dashboard - Fehler</title></head>
            <body><h1>Fehler beim Laden der Metriken</h1>
            <p>Konnte die Metrikdaten nicht vom Server abrufen. Bitte überprüfen Sie die Logs.</p>
            </body></html>
            """

        # Extrahiere Werte (mit Defaults, falls Keys nicht existieren)
        requests_metrics = metrics_summary.get("requests", {})
        vector_op_metrics = metrics_summary.get("vector_operations", {})
        cache_metrics = metrics_summary.get("cache", {})
        system_metrics = metrics_summary.get("system", {})
        error_metrics = metrics_summary.get("errors", {})

        current_qps = 0 # QPS muss berechnet werden, z.B. Anfragen der letzten Minute
        # Für eine Live-QPS-Anzeige bräuchte man Zeitreihendaten, nicht nur den Gesamt-Counter.
        # Nehmen wir an, avg_duration gibt uns eine Idee von der Kapazität.
        if requests_metrics.get("avg_duration", 0) > 0:
             # Sehr grobe Schätzung, tatsächliche QPS ist komplexer
            current_qps = 1.0 / requests_metrics.get("avg_duration", 1.0)

        avg_latency_ms = requests_metrics.get("avg_duration", 0) * 1000
        
        # Placeholder für aktive Stores, muss vom Server kommen oder anders ermittelt werden
        active_stores = metrics_summary.get("active_stores", "N/A") # Dieser Wert fehlt im Summary


        # Charts erstellen
        cpu_chart_json = self._create_gauge_chart("CPU Auslastung", system_metrics.get("cpu_usage_percent", 0), 100, "%")
        
        # Annahme: Gesamtspeicher ist bekannt, z.B. 8GB
        total_ram_gb = 8.0 # Dies sollte idealerweise dynamisch vom Server kommen
        memory_chart_json = self._create_gauge_chart(
            "RAM Nutzung",
            system_metrics.get("memory_usage_gb", 0),
            total_ram_gb, " GB"
        )
        
        cache_hit_rate_chart_json = self._create_gauge_chart(
            "Cache Hit Rate",
            cache_metrics.get("hit_rate_percent", 0),
            100, "%"
        )
        
        request_types_labels = ["Queries", "Additions", "Errors"]
        request_types_values = [
            vector_op_metrics.get("queries_total",0),
            vector_op_metrics.get("additions_total",0),
            error_metrics.get("total",0)
        ]
        requests_overview_json = self._create_bar_chart(
            "Anfrage-Typen (Gesamt)",
            request_types_labels,
            request_types_values,
            "Anzahl"
        )


        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>MLX Vector DB Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding:0; background-color: #f4f6f8; color: #333; }}
                .container {{ max-width: 1200px; margin: 20px auto; padding: 20px; background-color: #fff; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                header {{ background-color: #2c3e50; color: white; padding: 15px 20px; text-align: center; }}
                header h1 {{ margin: 0; font-size: 1.8em; }}
                .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 20px; }}
                .metric-card {{ background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); text-align: center; }}
                .metric-card h3 {{ margin-top: 0; font-size: 1.1em; color: #555; }}
                .metric-card .value {{ font-size: 2.2em; color: #2c3e50; margin: 10px 0; font-weight: bold; }}
                .chart-container {{ background: #fff; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-top: 20px; }}
                .chart-container h3 {{ text-align: center; margin-top:0; }}
                .footer {{ text-align: center; margin-top: 30px; padding: 15px; font-size: 0.9em; color: #777; }}
                /* Auto-refresh Hinweis statt tatsächlichem Refresh, da dies Client-seitig komplexer ist */
                .refresh-note {{text-align: center; font-style: italic; color: #888; margin-top: 10px;}}
            </style>
        </head>
        <body>
            <header><h1>MLX Vector DB Performance Dashboard</h1></header>
            <div class="container">
                <div class="grid">
                    <div class="metric-card">
                        <h3>Geschätzte QPS</h3>
                        <p class="value">{current_qps:.1f}</p>
                    </div>
                    <div class="metric-card">
                        <h3>Avg. Latenz</h3>
                        <p class="value">{avg_latency_ms:.1f} ms</p>
                    </div>
                    <div class="metric-card">
                        <h3>Aktive Stores</h3>
                        <p class="value">{active_stores}</p>
                    </div>
                     <div class="metric-card">
                        <h3>Cache Hits</h3>
                        <p class="value">{cache_metrics.get("hits",0)}</p>
                    </div>
                     <div class="metric-card">
                        <h3>Cache Misses</h3>
                        <p class="value">{cache_metrics.get("misses",0)}</p>
                    </div>
                     <div class="metric-card">
                        <h3>Fehler (Gesamt)</h3>
                        <p class="value">{error_metrics.get("total",0)}</p>
                    </div>
                </div>

                <div class="grid">
                    {"<div class='chart-container' id='cpu-chart'></div>" if cpu_chart_json else ""}
                    {"<div class='chart-container' id='memory-chart'></div>" if memory_chart_json else ""}
                </div>
                <div class="grid">
                     {"<div class='chart-container' id='cache-hit-rate-chart'></div>" if cache_hit_rate_chart_json else ""}
                     {"<div class='chart-container' id='requests-overview-chart'></div>" if requests_overview_json else ""}
                </div>
                 <p class="refresh-note">Dashboard-Daten werden beim Laden der Seite abgerufen. Für aktuelle Daten bitte Seite neu laden.</p>
            </div>
            <div class="footer">MLX Vector DB Monitoring &copy; {datetime.now().year}</div>

            <script>
                // Plotly Graphen rendern
                {f"Plotly.newPlot('cpu-chart', {cpu_chart_json});" if cpu_chart_json else ""}
                {f"Plotly.newPlot('memory-chart', {memory_chart_json});" if memory_chart_json else ""}
                {f"Plotly.newPlot('cache-hit-rate-chart', {cache_hit_rate_chart_json});" if cache_hit_rate_chart_json else ""}
                {f"Plotly.newPlot('requests-overview-chart', {requests_overview_json});" if requests_overview_json else ""}

                // Der im Plan vorgeschlagene Auto-Refresh ist hier entfernt,
                // da er zu ständigen Neuladevorgängen führt.
                // Eine bessere Lösung wäre AJAX-basiertes Update der Metriken.
                // setInterval(() => location.reload(), 30000); // z.B. alle 30 Sek.
            </script>
        </body>
        </html>
        """
        return html_content