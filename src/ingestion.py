import os
import time
import json
from pathlib import Path
from typing import Optional, List, Dict

# Logging
import src.logger_config
from loguru import logger

# Metrics
from prometheus_client import Counter, Histogram

# LangChain loaders
from langchain_community.document_loaders import TextLoader, PyPDFLoader, PyPDFium2Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embedding / model tools
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# Vector DB
import chromadb

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# --- Prometheus Metrics ---
METRICS = {}


def get_or_create_metric(name, metric_type, documentation, labels=None):
    """Get or create a Prometheus metric to avoid duplicate registration."""
    if name not in METRICS:
        if metric_type == 'counter':
            METRICS[name] = Counter(name, documentation, labels or [])
        elif metric_type == 'histogram':
            METRICS[name] = Histogram(name, documentation, labels or [])
    return METRICS[name]


INGESTION_COUNTER = get_or_create_metric(
    "ingestion_items_total", "counter", "Number of items ingested"
)
INGESTION_LATENCY = get_or_create_metric(
    "ingestion_latency_seconds", "histogram", "Latency of ingestion operations"
)

# --- Config ---
CHROMA_DIR = os.getenv("CHROMA_DIR", "./data/chroma")
TEXT_COL = os.getenv("TEXT_COLLECTION", "text_embeddings")
IMAGE_COL = os.getenv("IMAGE_COLLECTION", "image_embeddings")
TEXT_EMBED_MODEL = os.getenv("TEXT_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")

# MLP Model Path - use absolute path relative to this file
MLP_MODEL_PATH = os.getenv(
    "MLP_MODEL_PATH",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "mlp_projection.pt")
)

# --- Initialize Services ---
logger.info("Initializing ChromaDB at {}", CHROMA_DIR)
os.makedirs(CHROMA_DIR, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
text_collection = chroma_client.get_or_create_collection(TEXT_COL)
image_collection = chroma_client.get_or_create_collection(IMAGE_COL)

# Load Text Embedder (MiniLM 384D)
logger.info("Loading text embedder: {}", TEXT_EMBED_MODEL)
text_embedder = SentenceTransformer(TEXT_EMBED_MODEL)
TEXT_DIM = text_embedder.get_sentence_embedding_dimension()
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Using device: {}, Text embedding dimension: {}", device, TEXT_DIM)

# Load CLIP model
logger.info("Loading CLIP model: {}", CLIP_MODEL_NAME)
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
clip_proc = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
clip_model.eval()


# Projection MLP definition
class ProjectionMLP(nn.Module):
    """MLP to project CLIP image embeddings to text embedding space."""
    
    def __init__(self, in_dim: int = 512, out_dim: int = 384, hidden: int = 512):
        """
        Initialize the ProjectionMLP model.

        Args:
            in_dim (int): Input dimension (default: 512)
            out_dim (int): Output dimension (default: 384)
            hidden (int): Hidden dimension (default: 512)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        """
        Forward pass through the ProjectionMLP model.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        return self.net(x)


# Load projection MLP
projection_model = ProjectionMLP(in_dim=512, out_dim=TEXT_DIM).to(device)

if not os.path.exists(MLP_MODEL_PATH):
    logger.warning(
        "Projection model not found at {}. Image embeddings will use raw CLIP vectors.",
        MLP_MODEL_PATH
    )
    USE_PROJECTION = False
else:
    projection_model.load_state_dict(torch.load(MLP_MODEL_PATH, map_location=device))
    projection_model.eval()
    USE_PROJECTION = True
    logger.info("Loaded projection model from {} with output_dim={}", MLP_MODEL_PATH, TEXT_DIM)


# --- File Readers, Chunks, Embedding Functions ---

def read_text_file(path: str) -> str:
    """Read text content from various file formats."""
    p = Path(path)
    suffix = p.suffix.lower()
    
    if suffix in (".txt", ".md"):
        return p.read_text(encoding="utf-8")

    if suffix == ".pdf":
        try:
            loader = PyPDFLoader(str(p))
            docs = loader.load()
            if docs and docs[0].page_content.strip():
                return "\n\n".join([d.page_content for d in docs])
        except Exception as e:
            logger.warning(f"PyPDFLoader failed, trying PyPDFium2Loader. Error: {e}")
        
        try:
            loader = PyPDFium2Loader(str(p))
            docs = loader.load()
            return "\n\n".join([d.page_content for d in docs])
        except Exception as e:
            raise ValueError(f"PDF Load error: {e}")

    # Fallback to TextLoader
    try:
        loader = TextLoader(str(p))
        docs = loader.load()
        return "\n\n".join([d.page_content for d in docs])
    except Exception as e:
        raise ValueError(f"Could not load file: {e}")


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> List[str]:
    """Split text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)


def embed_text(texts: List[str]) -> np.ndarray:
    """Generate embeddings for text chunks."""
    return text_embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def embed_image(path: str) -> np.ndarray:
    """
    Generate embedding for an image.
    Projects CLIP image vector into text space if projection model is available.
    """
    image = Image.open(path).convert("RGB")
    inputs = clip_proc(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        feats = clip_model.get_image_features(**inputs)  # (1, 512)
        
        if USE_PROJECTION:
            projected = projection_model(feats)  # (1, 384)
            result = projected.cpu().numpy().reshape(-1)
        else:
            # If no projection model, use raw CLIP features
            result = feats.cpu().numpy().reshape(-1)

    return result


# --- Ingest API Logic ---

@INGESTION_LATENCY.time()
def upload_and_ingest(file_path: str, metadata: Optional[dict] = None, original_filename: Optional[str] = None) -> Dict:
    """
    Ingest a file (text, PDF, or image) into the vector database.
    
    Args:
        file_path: Path to the file to ingest
        metadata: Optional metadata dictionary
        original_filename: file name of source pdf
        
    Returns:
        Dictionary with ingestion results
    """
    start = time.time()
    metadata = metadata or {}
    p = Path(file_path)
    suffix = p.suffix.lower()
    
    logger.info("Ingesting {} (suffix={})", file_path, suffix)

    if suffix in (".txt", ".md", ".pdf", ".html", ".htm"):
        return _ingest_text_file(p, suffix, metadata, original_filename)
    elif suffix in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"):
        return _ingest_image_file(p, metadata, original_filename)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def _ingest_text_file(p: Path, suffix: str, metadata: dict, original_filename: Optional[str]=None) -> Dict:
    """Ingest a text-based file."""
    logger.info("Processing as Text/PDF")
    
    try:
        text = read_text_file(str(p))
    except Exception as e:
        logger.error("Failed to read file {}: {}", p, e)
        raise
    
    chunks = chunk_text(text)
    
    if not chunks:
        logger.warning("No text chunks found in '{}'", p)
        return {"inserted": 0, "type": "text", "error": "No content extracted"}
    
    vecs = embed_text(chunks)

    #ids = [f"text::{p.stem}::{i}" for i in range(len(chunks))]

    base_stem = Path(original_filename).stem if original_filename else p.stem
    ids = [f"text::{base_stem}::{i}" for i in range(len(chunks))]

   # metas = [{**metadata, "source": str(p), "chunk_index": i} for i in range(len(chunks))]
    source_name = original_filename or p.name  # e.g. "report.pdf"
    metas = [
        {
            **metadata,
            "source": source_name,      
            #"file_path": str(p),      
            "chunk_index": i,
        }
        for i in range(len(chunks))
    ]
    
    # Check for existing IDs and skip duplicates
    existing_ids = set()
    try:
        existing = text_collection.get(ids=ids)
        if existing and existing.get('ids'):
            existing_ids = set(existing['ids'])
    except Exception:
        pass
    
    new_ids = [id for id in ids if id not in existing_ids]
    if not new_ids:
        logger.info("All chunks already exist in database")
        return {"inserted": 0, "type": "text", "status": "already_exists"}
    
    # Filter to only new items
    new_indices = [i for i, id in enumerate(ids) if id not in existing_ids]
    new_chunks = [chunks[i] for i in new_indices]
    new_metas = [metas[i] for i in new_indices]
    new_vecs = [vecs[i].tolist() for i in new_indices]
    new_ids = [ids[i] for i in new_indices]
    
    text_collection.add(
        ids=new_ids, 
        documents=new_chunks, 
        metadatas=new_metas, 
        embeddings=new_vecs
    )
    
    INGESTION_COUNTER.inc(len(new_ids))
    logger.success("{} text chunks stored from {}", len(new_ids), p.name)
    
    return {"inserted": len(new_ids), "type": "text"}


def _ingest_image_file(p: Path, metadata: dict, original_filename: Optional[str]=None) -> Dict:
    """Ingest an image file."""
    logger.info("Processing as Image")
    
    try:
        img_vec = embed_image(str(p))
    except Exception as e:
        logger.error("Failed to embed image {}: {}", p, e)
        raise
    
    #doc_id = f"image::{p.stem}"
    base_stem = Path(original_filename).stem if original_filename else p.stem
    doc_id = f"image::{base_stem}"

    # Check if already exists
    try:
        existing = image_collection.get(ids=[doc_id])
        if existing and existing.get('ids'):
            logger.info("Image {} already exists in database", p.name)
            return {"inserted": 0, "type": "image", "status": "already_exists"}
    except Exception:
        pass
    
    embed_method = "clip+mlp_projection" if USE_PROJECTION else "clip_raw"
    embed_dim = TEXT_DIM if USE_PROJECTION else 512
    
    # Use original filename as the 'document' if available
    document_val = original_filename or str(p)
    source_name = original_filename or p.name

    image_collection.add(
        ids=[doc_id],
        documents=[document_val],  # this is what you'll see as "document" in Chroma
        metadatas=[{
            **metadata,
            "source": source_name,     # human filename
            #"file_path": str(p),       # temp path (for trace/debug)
            "embedding_dim": embed_dim,
            "embed_method": embed_method
        }],
        embeddings=[img_vec.tolist()]
    )
        
    INGESTION_COUNTER.inc()
    logger.success("Image '{}' embedded & stored with {}-dim ({})", p.name, embed_dim, embed_method)
    
    return {"inserted": 1, "type": "image"}


def get_collection_stats() -> Dict:
    """Get statistics about the collections."""
    return {
        "text_collection": {
            "name": TEXT_COL,
            "count": text_collection.count()
        },
        "image_collection": {
            "name": IMAGE_COL,
            "count": image_collection.count()
        }
    }