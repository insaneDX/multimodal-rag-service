"""
Main FastAPI application for Multimodal RAG Query Service.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import time
import re
import threading

import src.logger_config
from loguru import logger

from src.api import router as api_router
from src.retriever import hybrid_retrieve, build_grounded_prompt
from src.llm_utils import llm_generate, get_llm_backend, get_model_name
from src.self_check import extract_citations

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Prometheus monitoring
from src.monitoring import (
    QUERY_TOTAL, QUERY_ERRORS, CACHE_HITS, CACHE_MISSES,
    HALLUCINATION_COUNTER, RETRIEVAL_LATENCY, LLM_LATENCY, 
    QUERY_LATENCY, HALLUCINATION_RATE, start_metrics_server
)

# Start metrics server in background thread
threading.Thread(
    target=lambda: start_metrics_server(8001), 
    daemon=True
).start()


# ---- FastAPI App ----
app = FastAPI(
    title="Multimodal RAG Query Service",
    description="A RAG service supporting text and image retrieval with LLM generation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router)


# ---- Request/Response Models ----
class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str = Field(..., min_length=1, max_length=10000, description="The search query")
    top_k: int = Field(default=6, ge=1, le=50, description="Number of results to retrieve")
    use_reranker: bool = Field(default=False, description="Whether to use cross-encoder reranking")
    mode: str = Field(default="multimodal", description="Retrieval mode: text, image, or multimodal")


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str
    evidence: List[Dict[str, Any]]
    citations: List[str]
    latency_ms: float
    is_hallucinated: Optional[bool] = None


# ---- Cache ----
# Simple in-memory cache (replace with Redis in production)
_cache: Dict[str, Dict] = {}
MAX_CACHE_SIZE = 1000


def _get_cache_key(req: QueryRequest) -> str:
    """Generate cache key for a query request."""
    return f"{req.query}|{req.top_k}|{req.use_reranker}|{req.mode}"


# ---- Endpoints ----

@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "Multimodal RAG Query Service",
        "version": "1.0.0",
        "llm_backend": get_llm_backend(),
        "model": get_model_name()
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    """
    Process a RAG query: retrieve relevant documents and generate an answer.
    
    Args:
        req: Query request with search parameters
        
    Returns:
        Generated answer with evidence and citations
    """
    # Validate query
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    cache_key = _get_cache_key(req)
    QUERY_TOTAL.inc()
    start_all = time.time()
    
    try:
        # Check cache
        if cache_key in _cache:
            CACHE_HITS.inc()
            cached = _cache[cache_key].copy()
            cached['latency_ms'] = (time.time() - start_all) * 1000
            logger.info(f"Cache hit for query: {req.query[:50]}...")
            return QueryResponse(**cached)
        
        CACHE_MISSES.inc()

        # 1. Retrieval
        with RETRIEVAL_LATENCY.time():
            # Distribute top_k between text and image based on mode
            if req.mode == "text":
                results = hybrid_retrieve(
                    req.query, 
                    top_k_text=req.top_k, 
                    top_k_image=0, 
                    use_reranker=req.use_reranker
                )
            elif req.mode == "image":
                results = hybrid_retrieve(
                    req.query, 
                    top_k_text=0, 
                    top_k_image=req.top_k, 
                    use_reranker=req.use_reranker
                )
            else:  # multimodal
                text_k = max(1, req.top_k // 2)
                image_k = req.top_k - text_k
                results = hybrid_retrieve(
                    req.query, 
                    top_k_text=text_k, 
                    top_k_image=image_k, 
                    use_reranker=req.use_reranker
                )

        # 2. Build prompt
        prompt = build_grounded_prompt(req.query, results, top_k=req.top_k)

        # 3. LLM generation
        with LLM_LATENCY.time():
            answer = llm_generate(prompt, max_tokens=512)

        # 4. Extract citations and check for hallucination
        citations = extract_citations(answer)
        retrieved_ids = {r['id'] for r in results}
        
        # Check if citations are valid
        valid_citations = [c for c in citations if c in retrieved_ids]
        is_hallucinated = False
        
        if len(citations) == 0:
            # No citations at all - potential hallucination
            is_hallucinated = True
        elif len(valid_citations) == 0:
            # Citations present but none match retrieved docs
            is_hallucinated = True

        if is_hallucinated:
            HALLUCINATION_COUNTER.inc()

        # 5. Build response
        latency_ms = (time.time() - start_all) * 1000
        
        response_data = {
            "answer": answer,
            "evidence": results[:req.top_k],
            "citations": citations,
            "latency_ms": latency_ms,
            "is_hallucinated": is_hallucinated
        }
        
        resp = QueryResponse(**response_data)

        # 6. Update cache (with size limit)
        if len(_cache) >= MAX_CACHE_SIZE:
            # Remove oldest entry
            oldest_key = next(iter(_cache))
            del _cache[oldest_key]
        
        _cache[cache_key] = response_data
        
        logger.info(
            f"Query processed: '{req.query[:50]}...' | "
            f"Latency={latency_ms:.2f}ms | "
            f"Hallucinated={is_hallucinated} | "
            f"Citations={len(citations)}"
        )

        return resp

    except HTTPException:
        raise
    except Exception as e:
        QUERY_ERRORS.inc()
        logger.exception("Error processing query")
        raise HTTPException(status_code=500, detail=str(e))


# ---- DAG-based Query Endpoint ----
from src.langgraph_orchestrator import build_query_dag

# Build DAG once at startup
_query_dag = build_query_dag()


@app.post("/query_dag", response_model=QueryResponse)
async def query_dag_endpoint(req: QueryRequest):
    """
    Process a query using the DAG-based orchestrator.
    
    This endpoint provides more structured execution with retries
    and better observability.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    start_time = time.time()
    
    context = {
        "request": {
            "query": req.query,
            "mode": req.mode,
            "top_k": req.top_k,
            "use_reranker": req.use_reranker
        }
    }
    
    try:
        # Run DAG starting at 'retrieve'
        await _query_dag.run("retrieve", context)
        
        answer = context.get('llm_answer', '')
        citations = extract_citations(answer)
        retrieved = context.get('retrieved', [])
        verification = context.get('verification', {})
        
        latency_ms = (time.time() - start_time) * 1000
        
        return QueryResponse(
            answer=answer,
            evidence=retrieved[:req.top_k],
            citations=citations,
            latency_ms=latency_ms,
            is_hallucinated=verification.get('status') == 'flagged'
        )
        
    except Exception as e:
        logger.exception("DAG execution failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/cache")
async def clear_cache():
    """Clear the query cache."""
    global _cache
    count = len(_cache)
    _cache = {}
    logger.info(f"Cache cleared: {count} entries removed")
    return {"status": "ok", "cleared": count}