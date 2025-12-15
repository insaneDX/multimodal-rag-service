from prometheus_client import Counter, Histogram, Gauge, start_http_server, REGISTRY
from prometheus_client.core import CollectorRegistry
import threading

# Use a lock for thread safety
_metrics_lock = threading.Lock()
_metrics_server_started = False

# Counters
QUERY_TOTAL = Counter(
    "rag_query_total", 
    "Total RAG queries processed"
)
QUERY_ERRORS = Counter(
    "rag_query_errors_total", 
    "Total RAG queries that errored"
)
CACHE_HITS = Counter(
    "rag_cache_hits_total", 
    "Cache hits for queries"
)
CACHE_MISSES = Counter(
    "rag_cache_misses_total", 
    "Cache misses for queries"
)
HALLUCINATION_COUNTER = Counter(
    "rag_hallucination_total", 
    "Number of responses flagged as hallucinations"
)

# Histograms (latencies)
RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_latency_seconds", 
    "Latency of retrieval phase",
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)
LLM_LATENCY = Histogram(
    "rag_llm_latency_seconds", 
    "Latency of LLM inference",
    buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
)
QUERY_LATENCY = Histogram(
    "rag_query_latency_seconds", 
    "Latency end-to-end for queries",
    buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
)

# Gauges
HALLUCINATION_RATE = Gauge(
    "rag_hallucination_rate", 
    "Recent hallucination rate (client computed)"
)


def start_metrics_server(port: int = 8001):
    """Start Prometheus metrics server (thread-safe, only starts once)."""
    global _metrics_server_started
    with _metrics_lock:
        if not _metrics_server_started:
            try:
                start_http_server(port)
                _metrics_server_started = True
                print(f"Metrics server started on port {port}")
            except OSError as e:
                print(f"Could not start metrics server on port {port}: {e}")