from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import math
import os

import chromadb
from sentence_transformers import SentenceTransformer

# Logging setup
import src.logger_config
from loguru import logger

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Optional re-ranker
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    _has_reranker = True
except ImportError:
    _has_reranker = False
    logger.warning("Cross-encoder re-ranker dependencies not available")

# ----- Environment Config -----
CHROMA_DIR = os.getenv("CHROMA_DIR", "./data/chroma")
TEXT_COLLECTION = os.getenv("TEXT_COLLECTION", "text_embeddings")
IMAGE_COLLECTION = os.getenv("IMAGE_COLLECTION", "image_embeddings")
TEXT_EMBED_MODEL = os.getenv("TEXT_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER", "")

# ----- Initialize vector DB & models -----
logger.info("Initializing retriever with ChromaDB at {}", CHROMA_DIR)
os.makedirs(CHROMA_DIR, exist_ok=True)

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

# Use get_or_create_collection to avoid errors if collection doesn't exist
text_col = chroma_client.get_or_create_collection(TEXT_COLLECTION)
image_col = chroma_client.get_or_create_collection(IMAGE_COLLECTION)

logger.info("Loading text embedder: {}", TEXT_EMBED_MODEL)
embedder = SentenceTransformer(TEXT_EMBED_MODEL)

# Optional reranker initialization
re_tokenizer = None
re_model = None

if CROSS_ENCODER_MODEL and _has_reranker:
    try:
        re_tokenizer = AutoTokenizer.from_pretrained(CROSS_ENCODER_MODEL)
        re_model = AutoModelForSequenceClassification.from_pretrained(CROSS_ENCODER_MODEL)
        re_model.eval()
        if torch.cuda.is_available():
            re_model = re_model.to("cuda")
        logger.info("Cross-encoder re-ranker loaded: {}", CROSS_ENCODER_MODEL)
    except Exception as e:
        logger.warning("Failed to load cross-encoder model: {}", e)
        re_tokenizer = None
        re_model = None
elif CROSS_ENCODER_MODEL:
    logger.warning("Cross-encoder model specified but dependencies not available")


# ------------ Internal Utilities ---------------

def _norm_scores(distances: List[float], distance_type: str = "cosine") -> List[float]:
    """Normalize distance scores to similarity scores in [0, 1] range."""
    if not distances:
        return []
    
    arr = np.array(distances, dtype=float)
    
    # Convert cosine distance to similarity (distance = 1 - similarity for cosine)
    sim = 1.0 - arr
    
    minv, maxv = float(sim.min()), float(sim.max())
    
    if math.isclose(maxv, minv):
        return [1.0 for _ in sim]
    
    normed = ((sim - minv) / (maxv - minv)).tolist()
    logger.debug("Normalized scores: min={:.4f}, max={:.4f}", minv, maxv)
    
    return normed


def _format_chroma_result(raw: Optional[Dict]) -> List[Dict]:
    """Format ChromaDB query results into a consistent structure."""
    if not raw:
        return []
    
    ids = raw.get("ids", [[]])[0] if raw.get("ids") else []
    docs = raw.get("documents", [[]])[0] if raw.get("documents") else []
    metas = raw.get("metadatas", [[]])[0] if raw.get("metadatas") else []
    dists = raw.get("distances", [[]])[0] if raw.get("distances") else []
    
    results = []
    for i, _id in enumerate(ids):
        results.append({
            "id": _id,
            "document": docs[i] if i < len(docs) else "",
            "metadata": metas[i] if i < len(metas) else {},
            "distance": dists[i] if i < len(dists) else 1.0
        })
    
    return results


def _fuse_results(
    text_hits: List[Dict], 
    image_hits: List[Dict], 
    w_text: float = 0.7, 
    w_image: float = 0.3
) -> List[Dict]:
    """Fuse text and image results with weighted scoring."""
    fused = []
    
    for t in text_hits:
        fused.append({
            **t, 
            "fused_score": t.get("score", 0.0) * w_text, 
            "modal": "text"
        })
    
    for im in image_hits:
        fused.append({
            **im, 
            "fused_score": im.get("score", 0.0) * w_image, 
            "modal": "image"
        })
    
    return sorted(fused, key=lambda x: x["fused_score"], reverse=True)


def _cross_encoder_score(pairs: List[Tuple[str, str]]) -> List[float]:
    """Compute cross-encoder scores for query-document pairs."""
    if not re_tokenizer or not re_model:
        raise RuntimeError("Cross-encoder not configured.")

    texts = [f"{q} [SEP] {c}" for q, c in pairs]
    inputs = re_tokenizer(
        texts, 
        padding=True, 
        truncation=True, 
        max_length=512,
        return_tensors="pt"
    )
    
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    with torch.no_grad():
        out = re_model(**inputs)
        logits = out.logits.squeeze(-1).cpu().numpy()
    
    # Ensure logits is 1D
    if logits.ndim == 0:
        logits = np.array([logits])
    
    minv, maxv = float(logits.min()), float(logits.max())
    if math.isclose(maxv, minv):
        return [1.0 for _ in logits]
    
    normed = ((logits - minv) / (maxv - minv)).tolist()
    logger.debug("Reranker scores normalized: {}", normed[:5])
    
    return normed


# ------------ Main Retrieval Function ------------

def hybrid_retrieve(
    query: str,
    top_k_text: int = 5,
    top_k_image: int = 5,
    weight_text: float = 0.7,
    weight_image: float = 0.3,
    use_reranker: bool = False
) -> List[Dict[str, Any]]:
    """
    Perform hybrid retrieval across text and image collections.
    
    Args:
        query: Search query string
        top_k_text: Number of text results to retrieve
        top_k_image: Number of image results to retrieve
        weight_text: Weight for text results in fusion
        weight_image: Weight for image results in fusion
        use_reranker: Whether to apply cross-encoder reranking
        
    Returns:
        List of retrieved documents with scores
    """
    logger.info("Starting hybrid retrieval: query='{}' (text_k={}, img_k={})", 
                query[:50], top_k_text, top_k_image)

    # Generate query embedding
    qvec = embedder.encode([query], convert_to_numpy=True)[0]

    text_hits = []
    image_hits = []

    # ---- Text Retrieval ----
    if top_k_text > 0:
        try:
            text_raw = text_col.query(
                query_embeddings=[qvec.tolist()],
                n_results=top_k_text,
                include=["documents", "metadatas", "distances"]
            )
            text_hits = _format_chroma_result(text_raw)
            text_scores = _norm_scores([hit["distance"] for hit in text_hits])
            for i, score in enumerate(text_scores):
                text_hits[i]["score"] = score
            logger.info("Text results: {} retrieved", len(text_hits))
        except Exception as e:
            logger.warning("Text retrieval failed: {}", e)

    # ---- Image Retrieval ----
    if top_k_image > 0:
        try:
            image_raw = image_col.query(
                query_embeddings=[qvec.tolist()],
                n_results=top_k_image,
                include=["documents", "metadatas", "distances"]
            )
            image_hits = _format_chroma_result(image_raw)
            image_scores = _norm_scores([hit["distance"] for hit in image_hits])
            for i, score in enumerate(image_scores):
                image_hits[i]["score"] = score
            logger.info("Image results: {} retrieved", len(image_hits))
        except Exception as e:
            logger.warning("Image retrieval failed: {}", e)

    # ---- Fusion ----
    fused = _fuse_results(text_hits, image_hits, weight_text, weight_image)
    logger.info("Total fused results: {}", len(fused))

    # ---- Optional Reranker ----
    if use_reranker and re_model and fused:
        logger.info("Applying cross-encoder re-ranker...")
        try:
            pairs = []
            for item in fused:
                if item["modal"] == "text":
                    doc_text = item["document"]
                else:
                    doc_text = item["metadata"].get("caption") or item["document"]
                pairs.append((query, doc_text))
            
            rerank_scores = _cross_encoder_score(pairs)
            
            for i, score in enumerate(rerank_scores):
                # Combine original fused score with reranker score
                fused[i]["fused_score"] = 0.5 * fused[i]["fused_score"] + 0.5 * score
            
            fused = sorted(fused, key=lambda x: x["fused_score"], reverse=True)
            logger.info("Reranking complete.")
        except Exception as e:
            logger.warning("Reranking failed: {}", e)
    elif use_reranker and not re_model:
        logger.warning("Re-ranker requested but not available.")

    return fused


# ------------ Prompt Builder ------------

PROMPT_TEMPLATE = """
You are a helpful, cautious assistant. Use only the evidence provided below to answer the user's question.
Cite sources with [source_id] after each statement that uses that source.

User Question:
{query}

Evidence (top {k}):
{evidence_blocks}

Instructions:
- Use only the evidence provided above. Do not use outside knowledge.
- If the evidence does not support a claim, say "I cannot verify this from the provided documents."
- Be concise and accurate.
- Always cite your sources using [source_id] format.
- If the evidence is insufficient, explain what additional information would be needed.
"""


def build_grounded_prompt(query: str, retrieved: List[Dict], top_k: int = 5) -> str:
    """
    Build a grounded prompt with retrieved evidence for LLM generation.
    
    Args:
        query: User's question
        retrieved: List of retrieved documents
        top_k: Maximum number of evidence items to include
        
    Returns:
        Formatted prompt string
    """
    logger.info("Building grounded prompt with top_k={}", top_k)
    
    lines = []
    for i, r in enumerate(retrieved[:top_k]):
        sid = r['id']
        modal = r.get('modal', 'text')
        src = r.get('metadata', {}).get('source', 'unknown')
        score = r.get('fused_score', r.get('score', 0.0))
        
        if modal == 'text':
            excerpt = r.get('document', '')[:800]
            lines.append(
                f"[{sid}] (text, relevance={score:.3f}) - source: {src}\n{excerpt}"
            )
        else:
            caption = r.get('metadata', {}).get('caption', '(no caption)')
            image_path = r.get('document', '')
            lines.append(
                f"[{sid}] (image, relevance={score:.3f}) - source: {src}\n"
                f"Caption: {caption}\nImage Path: {image_path}"
            )
    
    if not lines:
        evidence_blocks = "(No relevant evidence found)"
    else:
        evidence_blocks = "\n\n---\n\n".join(lines)
    
    prompt = PROMPT_TEMPLATE.format(
        query=query, 
        k=min(top_k, len(retrieved)), 
        evidence_blocks=evidence_blocks
    )
    
    return prompt