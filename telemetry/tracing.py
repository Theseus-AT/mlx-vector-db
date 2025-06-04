# Neue Datei: telemetry/tracing_config.py
# Konfiguriert OpenTelemetry Tracing.

# MLX Specificity: Das Tracing von MLX-Operationen erfordert ggf. manuelle Spans,
#                  da automatische Instrumentierung für MLX noch nicht Standard ist.
#                  Ein benutzerdefinierter MLXInstrumentor (wie im Plan angedeutet)
#                  wäre hier ideal.
# LLM Anbindung: Tracing hilft, den gesamten Anfragefluss von einer LLM-Anwendung
#                durch die Vektor-DB bis hin zu einzelnen MLX-Operationen zu verfolgen
#                und Performance-Bottlenecks zu identifizieren.

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter # ConsoleExporter für Debugging
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter # OTLP Exporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource # Für Service-Namen etc.

# MLX-spezifische Imports und Hilfsfunktionen (Plan-Annahme)
import mlx.core as mx
import time
from contextlib import contextmanager
import os
import logging

logger = logging.getLogger("mlx_vector_db.telemetry")

# Globale Tracer-Instanz
tracer: Optional[trace.Tracer] = None

def setup_tracing(app, service_name: str = "mlx-vector-db", enable_console_exporter: bool = False):
    """Konfiguriert OpenTelemetry Tracing für die Anwendung."""
    global tracer

    if not os.getenv("ENABLE_TRACING", "false").lower() == "true":
        logger.info("OpenTelemetry Tracing is disabled via ENABLE_TRACING flag.")
        return

    otlp_endpoint = os.getenv("OTLP_ENDPOINT", "http://localhost:4317") # Gemäß Plan .env
    otlp_insecure = os.getenv("OTLP_INSECURE", "true").lower() == "true"

    resource = Resource(attributes={
        "service.name": service_name,
        "service.version": "1.0.0" # Sollte dynamisch sein
    })

    # Tracer Provider setzen
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)
    
    tracer = trace.get_tracer(__name__) # Initialisiert den globalen Tracer

    # OTLP Exporter konfigurieren
    otlp_exporter = OTLPSpanExporter(
        endpoint=otlp_endpoint,
        insecure=otlp_insecure # Für lokale Tests oft true, in Produktion false mit TLS
    )
    
    # Span Processor hinzufügen (Batch für bessere Performance)
    otlp_processor = BatchSpanProcessor(otlp_exporter)
    provider.add_span_processor(otlp_processor)
    
    logger.info(f"OpenTelemetry OTLP exporter configured for endpoint: {otlp_endpoint}, insecure: {otlp_insecure}")

    if enable_console_exporter or os.getenv("OTEL_CONSOLE_EXPORTER", "false").lower() == "true":
        console_exporter = ConsoleSpanExporter()
        console_processor = BatchSpanProcessor(console_exporter)
        provider.add_span_processor(console_processor)
        logger.info("OpenTelemetry ConsoleSpanExporter enabled.")

    # FastAPI automatisch instrumentieren
    FastAPIInstrumentor.instrument_app(app)
    logger.info("FastAPIInstrumentor enabled.")
    
    # Hier könnte die Registrierung des benutzerdefinierten MLXInstrumentor erfolgen,
    # falls dieser existiert. Der Plan erwähnt `MLXInstrumentor`, was aber
    # wahrscheinlich eine Eigenentwicklung wäre, da es (noch) keine offizielle gibt.
    # MLXInstrumentor().instrument() # Beispielhafter Aufruf

    logger.info("OpenTelemetry tracing setup complete.")


# Kontextmanager für manuelles Tracing von MLX-Operationen (aus dem Plan)
@contextmanager
def trace_mlx_operation(operation_name: str, attributes: Optional[Dict[str, Any]] = None):
    """
    Ein Kontextmanager zum Erstellen eines Spans für eine MLX-Operation.
    Stellt sicher, dass der globale `tracer` initialisiert wurde via `setup_tracing`.
    """
    if tracer is None: # Wenn Tracing nicht initialisiert wurde, nichts tun
        yield None
        return

    with tracer.start_as_current_span(operation_name) as span:
        span.set_attribute("mlx.operation", operation_name)
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        
        start_op_time = time.perf_counter()
        try:
            yield span
        finally:
            # MLX-Operationen sind oft lazy. Ein mx.eval() ist hier ggf. nötig,
            # um die Ausführungszeit korrekt zu messen, falls das Ergebnis des
            # Yield-Blocks ein unevaluiertes mx.array ist. Dies muss im aufrufenden
            # Code bedacht werden.
            # Besser ist, die Dauer direkt in der Operation zu messen, wo eval() stattfindet.
            duration_op_ms = (time.perf_counter() - start_op_time) * 1000
            span.set_attribute("mlx.duration_ms", duration_op_ms)
            # Ggf. weitere MLX-spezifische Attribute nach der Operation setzen.

# Verwendung von trace_mlx_operation:
# from .telemetry.tracing_config import trace_mlx_operation
#
# def some_mlx_heavy_function(data_mx: mx.array):
#     with trace_mlx_operation("my_custom_mlx_op", attributes={"input_shape": str(data_mx.shape)}):
#         result_mx = data_mx * 2 + 5 # Beispiel MLX Operation
#         mx.eval(result_mx) # Wichtig für korrekte Zeitmessung, falls nicht schon vorher evaluiert
#         # span.set_attribute("output_shape", str(result_mx.shape)) # Zugriff auf span Objekt möglich
#         return result_mx