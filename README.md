# Multimodal RAG System

A production-ready Retrieval-Augmented Generation (RAG) system that supports both text and image modalities, featuring automatic hallucination detection, citation validation, and a user-friendly web interface.

## Features

- **Multimodal Support**: Index and retrieve from text documents (TXT, PDF, MD) and images (PNG, JPG, WEBP, BMP)
- **Hybrid Retrieval**: Combines text and image embeddings for comprehensive search
- **Grounded Generation**: LLM responses are grounded in retrieved evidence with citations
- **Hallucination Detection**: Automatic verification of generated answers against source documents
- **Cross-Encoder Reranking**: Optional reranking for improved relevance
- **DAG-based Orchestration**: Structured pipeline execution with retry logic
- **Production Ready**: Prometheus metrics, health checks, and caching
- **Web Interface**: Streamlit-based UI for easy interaction

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Streamlit UI  │────▶│   FastAPI API  │────▶│   ChromaDB      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │                          │
                               ▼                          ▼
                        ┌─────────────┐           ┌──────────────┐
                        │  LLM (Groq) │           │  Embeddings  │
                        └─────────────┘           │  - MiniLM    │
                                                  │  - CLIP      │
                                                  └──────────────┘
```

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, for faster inference)
- API keys for LLM providers (Groq, OpenAI)

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/multimodal-rag.git
cd multimodal-rag
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a `.env` file in the project root:

```env
# LLM Configuration
LLM_BACKEND=groq  # Options: groq, openai, hf
LLM_MODEL=llama-3.1-70b-versatile
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key  # If using OpenAI

# Vector Database
CHROMA_DIR=./data/chroma
TEXT_COLLECTION=text_embeddings
IMAGE_COLLECTION=image_embeddings

# Embedding Models
TEXT_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
CLIP_MODEL=openai/clip-vit-base-patch32

# Optional: Cross-encoder for reranking
CROSS_ENCODER=cross-encoder/ms-marco-MiniLM-L-6-v2

# API Configuration
API_BASE_URL=http://localhost:8000
```

### 4. Train the Projection Model (Optional)

If you want to use the MLP projection for better image-text alignment:

```bash
python src/train_projection.py
```

This will create `mlp_projection.pt` in the project root.

### 5. Start the API Server

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Launch the Web Interface

In a new terminal:

```bash
streamlit run ui/streamlit_app.py
```

Access the UI at `http://localhost:8501`

## Usage

### Via Web Interface

1. **Upload Documents**: Navigate to "Upload Documents" and upload your files
2. **Query**: Go to "Query" page and ask questions about your documents
3. **Review**: Check the generated answer, citations, and hallucination status

### Via API

#### Upload a Document

```bash
curl -X POST "http://localhost:8000/api/v1/upload" \
  -F "file=@document.pdf" \
  -F 'metadata={"source": "research", "date": "2024"}'
```

#### Query the System

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main findings?",
    "top_k": 5,
    "mode": "multimodal",
    "use_reranker": false
  }'
```

#### Get Statistics

```bash
curl "http://localhost:8000/api/v1/stats"
```

## Configuration

### Retrieval Modes

- **multimodal**: Search both text and images (default)
- **text**: Search only text documents
- **image**: Search only images

### Query Parameters

- `top_k`: Number of documents to retrieve (1-20)
- `use_reranker`: Enable cross-encoder reranking for better relevance
- `use_dag`: Use DAG-based orchestrator for structured execution

## Monitoring

### Prometheus Metrics

Metrics are exposed at `http://localhost:8001/metrics`:

- `rag_query_total`: Total queries processed
- `rag_query_errors_total`: Failed queries
- `rag_retrieval_latency_seconds`: Retrieval phase latency
- `rag_llm_latency_seconds`: LLM generation latency
- `rag_hallucination_total`: Responses flagged as hallucinations

### Health Checks

- API Health: `http://localhost:8000/health`
- Detailed Stats: `http://localhost:8000/api/v1/stats`

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Evaluate retrieval performance:

```bash
python tests/retrieval_eval.py --dataset queries.json --k 10
```

## Project Structure

```
multimodal-rag/
├── src/
│   ├── main.py              # FastAPI application
│   ├── api.py               # API endpoints
│   ├── ingestion.py         # Document ingestion logic
│   ├── retriever.py         # Retrieval and ranking
│   ├── llm_utils.py         # LLM integration
│   ├── self_check.py        # Hallucination detection
│   ├── langgraph_orchestrator.py  # DAG pipeline
│   ├── monitoring.py        # Prometheus metrics
│   └── train_projection.py  # Train image-text projection
├── ui/
│   └── streamlit_app.py     # Web interface
├── tests/
│   ├── test_retriever.py    # Unit tests
│   └── retrieval_eval.py    # Evaluation script
├── data/
│   └── chroma/              # Vector database storage
├── requirements.txt
├── .env.example
└── README.md
```

## Advanced Features

### Custom Embeddings

To use custom embedding models, modify the environment variables:

```env
TEXT_EMBED_MODEL=your-custom-model
CLIP_MODEL=your-clip-variant
```

### Hallucination Detection

The system automatically checks if:
- Citations reference actual retrieved documents
- Answer content is grounded in the evidence
- Keywords in the answer appear in source documents

### DAG-based Orchestration

The DAG pipeline provides:
- Structured execution flow
- Automatic retries for failed steps
- Better observability
- Modular architecture

---

**Note**: This is a research/educational project. For production use, ensure proper security, authentication, and rate limiting are implemented.
