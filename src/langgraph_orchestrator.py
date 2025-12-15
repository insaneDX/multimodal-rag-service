"""
DAG-based orchestration for RAG pipeline.
"""
import asyncio
import time
from typing import Callable, Dict, Any, List, Optional

import src.logger_config
from loguru import logger

from src.monitoring import QUERY_TOTAL
from src.ingestion import upload_and_ingest
from src.retriever import hybrid_retrieve, build_grounded_prompt
from src.llm_utils import llm_generate  # Import from separate module to avoid circular import
from src.self_check import verify_answer, validate_citations

# Type alias for node functions
NodeFunc = Callable[[Dict[str, Any]], Any]


# --- Orchestrator Framework ---

class Node:
    """Represents a node in the execution DAG."""
    
    def __init__(self, name: str, func: NodeFunc, retries: int = 0):
        self.name = name
        self.func = func
        self.retries = retries


class Orchestrator:
    """Simple DAG-based orchestrator for pipeline execution."""
    
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, List[str]] = {}

    def add_node(self, node: Node) -> "Orchestrator":
        """Add a node to the DAG."""
        self.nodes[node.name] = node
        self.edges.setdefault(node.name, [])
        return self

    def add_edge(self, from_node: str, to_node: str) -> "Orchestrator":
        """Add an edge between two nodes."""
        self.edges.setdefault(from_node, []).append(to_node)
        return self

    async def run(self, start_node: str, context: Dict[str, Any], stop_on_error: bool = True) -> Dict[str, Any]:
        """
        Execute the DAG starting from a given node.
        
        Args:
            start_node: Name of the starting node
            context: Shared context dictionary
            stop_on_error: Whether to stop execution on first error
            
        Returns:
            The updated context dictionary
        """
        queue = [start_node]
        visited = set()
        errors = []
        
        while queue:
            node_name = queue.pop(0)
            
            if node_name in visited:
                continue

            node = self.nodes.get(node_name)
            if node is None:
                error_msg = f"Node not found: {node_name}"
                logger.error(error_msg)
                if stop_on_error:
                    raise RuntimeError(error_msg)
                errors.append(error_msg)
                continue

            logger.info(f"Running node: '{node_name}'")
            start_time = time.time()
            success = False
            last_exc = None

            # Retry loop
            for attempt in range(node.retries + 1):
                try:
                    res = node.func(context)
                    if asyncio.iscoroutine(res):
                        await res
                    success = True
                    break
                except Exception as e:
                    last_exc = e
                    logger.warning(
                        f"Node '{node_name}' failed on attempt {attempt + 1}/{node.retries + 1}: {e}"
                    )
                    if attempt < node.retries:
                        await asyncio.sleep(0.5 * (attempt + 1))

            elapsed = time.time() - start_time
            
            if success:
                logger.info(f"Node '{node_name}' completed in {elapsed:.2f}s")
            else:
                logger.error(f"Node '{node_name}' failed after {node.retries + 1} attempts")
                if stop_on_error:
                    raise last_exc
                errors.append(f"{node_name}: {last_exc}")
                continue

            visited.add(node_name)
            
            # Add next nodes to queue
            next_nodes = self.edges.get(node_name, [])
            queue.extend(next_nodes)
        
        context["_orchestration_errors"] = errors
        return context


# --- DAG Node Functions ---

async def ingest_node(context: Dict[str, Any]):
    """Ingest a file into the vector database."""
    logger.info("ingest_node: Starting file ingestion...")
    
    path = context.get("file_path")
    if not path:
        raise ValueError("file_path not provided in context")
    
    meta = context.get("metadata", {})
    res = upload_and_ingest(path, metadata=meta)
    context["ingest_result"] = res
    
    logger.success("Ingested file with {} items", res.get("inserted", 0))


async def embed_index_node(context: Dict[str, Any]):
    """Placeholder for additional embedding/indexing logic."""
    logger.info("embed_index_node: (placeholder)")
    context["embed_index_result"] = {"status": "ok"}
    logger.success("embed_index_node done")


async def retrieve_node(context: Dict[str, Any]):
    """Retrieve relevant documents for a query."""
    logger.info("retrieve_node: Starting retrieval...")
    
    request = context.get("request", {})
    query = request.get("query")
    
    if not query:
        raise ValueError("query not provided in request")
    
    mode = request.get("mode", "multimodal")
    use_reranker = request.get("use_reranker", False)

    # Configure retrieval based on mode
    if mode == "text":
        res = hybrid_retrieve(
            query, 
            top_k_text=8, 
            top_k_image=0, 
            weight_text=1.0, 
            weight_image=0.0,
            use_reranker=use_reranker
        )
    elif mode == "image":
        res = hybrid_retrieve(
            query, 
            top_k_text=0, 
            top_k_image=8, 
            weight_text=0.0, 
            weight_image=1.0,
            use_reranker=use_reranker
        )
    else:  # multimodal
        res = hybrid_retrieve(
            query, 
            top_k_text=5, 
            top_k_image=3, 
            weight_text=0.7, 
            weight_image=0.3,
            use_reranker=use_reranker
        )

    context["retrieved"] = res
    logger.success("Retrieved {} documents (mode={})", len(res), mode)


async def prompt_build_node(context: Dict[str, Any]):
    """Build the grounded prompt for LLM generation."""
    logger.info("prompt_build_node: Building prompt...")
    
    query = context["request"]["query"]
    retrieved = context.get("retrieved", [])
    top_k = context.get("request", {}).get("top_k", 6)
    
    prompt = build_grounded_prompt(query, retrieved, top_k=top_k)
    context["prompt"] = prompt
    
    logger.success("Prompt built with {} evidence items", min(len(retrieved), top_k))


async def llm_node(context: Dict[str, Any]):
    """Generate answer using LLM."""
    logger.info("llm_node: Generating LLM answer...")
    
    prompt = context.get("prompt")
    if not prompt:
        raise ValueError("prompt not found in context")
    
    # Run LLM in thread pool to avoid blocking
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, llm_generate, prompt)
    
    context["llm_answer"] = result
    logger.success("LLM returned answer of {} chars", len(result))


async def verify_node(context: Dict[str, Any]):
    """Verify the LLM answer against retrieved evidence."""
    logger.info("verify_node: Starting verification...")
    
    answer = context.get("llm_answer", "")
    retrieved = context.get("retrieved", [])
    
    try:
        verification = verify_answer(answer, retrieved)
        citation_check = validate_citations(answer, retrieved)
        verification["citations"] = citation_check
    except Exception as e:
        logger.warning("Verification failed: {}", e)
        verification = {
            "status": "error",
            "issues": [str(e)],
            "score": 0.0,
            "notes": "Verification failed with an error"
        }
    
    context["verification"] = verification
    logger.success("Verification status: {}", verification.get("status"))


async def metrics_node(context: Dict[str, Any]):
    """Update Prometheus metrics."""
    logger.info("metrics_node: Updating counters...")
    
    try:
        QUERY_TOTAL.inc()
        logger.success("QUERY_TOTAL incremented")
    except Exception as e:
        logger.warning("Metrics update failed: {}", e)


# --- DAG Builder Method ---

def build_rag_dag() -> Orchestrator:
    """
    Build the RAG pipeline DAG.
    
    Returns:
        Configured Orchestrator instance
    """
    o = Orchestrator()

    # Add nodes with retry configuration
    o.add_node(Node("ingest", ingest_node, retries=1))
    o.add_node(Node("embed_index", embed_index_node, retries=0))
    o.add_node(Node("retrieve", retrieve_node, retries=2))
    o.add_node(Node("prompt", prompt_build_node, retries=0))
    o.add_node(Node("llm", llm_node, retries=2))
    o.add_node(Node("verify", verify_node, retries=0))
    o.add_node(Node("metrics", metrics_node, retries=0))

    # Define execution order
    o.add_edge("ingest", "embed_index")
    o.add_edge("embed_index", "retrieve")
    o.add_edge("retrieve", "prompt")
    o.add_edge("prompt", "llm")
    o.add_edge("llm", "verify")
    o.add_edge("verify", "metrics")

    logger.info("RAG DAG built with {} nodes", len(o.nodes))
    return o


def build_query_dag() -> Orchestrator:
    """
    Build a DAG for query-only operations (no ingestion).
    
    Returns:
        Configured Orchestrator instance for queries
    """
    o = Orchestrator()

    o.add_node(Node("retrieve", retrieve_node, retries=2))
    o.add_node(Node("prompt", prompt_build_node, retries=0))
    o.add_node(Node("llm", llm_node, retries=2))
    o.add_node(Node("verify", verify_node, retries=0))
    o.add_node(Node("metrics", metrics_node, retries=0))

    o.add_edge("retrieve", "prompt")
    o.add_edge("prompt", "llm")
    o.add_edge("llm", "verify")
    o.add_edge("verify", "metrics")

    logger.info("Query DAG built with {} nodes", len(o.nodes))
    return o