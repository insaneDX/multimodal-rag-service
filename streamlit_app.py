"""
Streamlit Frontend for Multimodal RAG Query Service.
"""
import streamlit as st
import requests
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any
import os
import threading

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# ---- Configuration ----
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_V1_URL = f"{API_BASE_URL}/api/v1"

# ---- Page Configuration ----
st.set_page_config(
    page_title="Multimodal RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# For streamlit deployment, we run the FastAPI app in a background thread
from src.app import app as fastapi_app

def run_backend():
    import uvicorn
    uvicorn.run(
        fastapi_app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

@st.cache_resource
def start_backend():
    thread = threading.Thread(target=run_backend, daemon=True)
    thread.start()
    time.sleep(2)
    return True

# start backend on first access
start_backend()



# ---- Custom CSS ----
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        background-color: #E8F5E9;
        border-left: 4px solid #4CAF50;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        background-color: #FFEBEE;
        border-left: 4px solid #F44336;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        background-color: #E3F2FD;
        border-left: 4px solid #2196F3;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .evidence-card {
        padding: 1rem;
        background-color: #F5F5F5;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #E0E0E0;
    }
    .metric-card {
        background-color: #FAFAFA;
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .citation-tag {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        background-color: #E3F2FD;
        color: #1565C0;
        border-radius: 4px;
        font-size: 0.85rem;
        margin: 0.2rem;
    }
    .hallucination-warning {
        padding: 0.5rem 1rem;
        background-color: #FFF3E0;
        border-left: 4px solid #FF9800;
        border-radius: 4px;
    }
    .hallucination-ok {
        padding: 0.5rem 1rem;
        background-color: #E8F5E9;
        border-left: 4px solid #4CAF50;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ---- Helper Functions ----

def check_api_health() -> bool:
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def get_api_info() -> Optional[Dict]:
    """Get API information."""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException:
        pass
    return None


def upload_file(file, metadata: Dict) -> Dict:
    """Upload a file to the API."""
    files = {"file": (file.name, file.getvalue(), file.type)}
    data = {"metadata": json.dumps(metadata)}
    
    response = requests.post(
        f"{API_V1_URL}/upload",
        files=files,
        data=data,
        timeout=60
    )
    return response.json() if response.status_code == 200 else {"error": response.text}


def query_rag(
    query: str, 
    top_k: int = 5, 
    mode: str = "multimodal",
    use_reranker: bool = False,
    use_dag: bool = False
) -> Dict:
    """
    Send a query to the RAG system.

    Args:
        query (str): The search query.
        top_k (int): The number of results to retrieve.
        mode (str): The retrieval mode: text, image, or multimodal.
        use_reranker (bool): Whether to use reranking.
        use_dag (bool): Whether to use the DAG-based orchestrator.

    Returns:
        Dict: A dictionary containing the response from the API.
    """
    endpoint = "/query_dag" if use_dag else "/query"
    
    payload = {
        "query": query,
        "top_k": top_k,
        "mode": mode,
        "use_reranker": use_reranker
    }
    
    # Send the request to the API
    response = requests.post(
        f"{API_BASE_URL}{endpoint}",
        json=payload,
        timeout=120
    )
    
    # Check if the request was successful
    if response.status_code == 200:
        return response.json()
    else:
        # Return an error message if the request failed
        return {"error": response.text, "status_code": response.status_code}


def get_stats() -> Dict:
    """Get collection statistics."""
    try:
        response = requests.get(f"{API_V1_URL}/stats", timeout=10)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}
    return {"error": "Failed to get stats"}


def clear_cache() -> Dict:
    """Clear the query cache."""
    try:
        response = requests.delete(f"{API_BASE_URL}/cache", timeout=10)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}
    return {"error": "Failed to clear cache"}


# ---- Sidebar ----

def render_sidebar():
    """Render the sidebar with navigation and settings."""
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/search-in-cloud.png", width=80)
        st.title("RAG System")
        
        # Navigation
        st.markdown("### Navigation")
        page = st.radio(
            "Select Page",
            ["🏠 Home", "📤 Upload Documents", "💬 Query", "📊 Statistics", "⚙️ Settings"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # API Status
        st.markdown("### API Status")
        if check_api_health():
            st.success("✅ API Online")
            api_info = get_api_info()
            if api_info:
                st.caption(f"Backend: {api_info.get('llm_backend', 'N/A')}")
                st.caption(f"Model: {api_info.get('model', 'N/A')[:30]}...")
        else:
            st.error("❌ API Offline")
            st.caption("Make sure the API is running on port 8000")
        
        st.markdown("---")
        
        # Quick Stats
        st.markdown("### Quick Stats")
        stats = get_stats()
        if "stats" in stats:
            text_count = stats["stats"].get("text_collection", {}).get("count", 0)
            image_count = stats["stats"].get("image_collection", {}).get("count", 0)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Texts", text_count)
            with col2:
                st.metric("Images", image_count)
        
        return page


# ---- Pages ----

def render_home():
    """Render the home page."""
    st.markdown('<h1 class="main-header">Multimodal RAG System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload documents, ask questions, get AI-powered answers with citations</p>', unsafe_allow_html=True)
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Upload</h3>
            <p>Upload text files, PDFs, and images to build your knowledge base</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Query</h3>
            <p>Ask questions and get answers grounded in your documents</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>✅ Verify</h3>
            <p>Automatic hallucination detection and citation validation</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Getting Started
    st.markdown("### Getting Started")
    
    st.markdown("""
    1. **Upload Documents**: Go to the Upload page and add your text files, PDFs, or images
    2. **Ask Questions**: Navigate to the Query page and ask questions about your documents
    3. **Review Answers**: Get AI-generated answers with citations and evidence
    """)
    
    # Supported formats
    st.markdown("### Supported File Formats")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Text Documents:**
        - `.txt` - Plain text files
        - `.md` - Markdown files
        - `.pdf` - PDF documents
        """)
    
    with col2:
        st.markdown("""
        **Images:**
        - `.png` - PNG images
        - `.jpg` / `.jpeg` - JPEG images
        - `.webp` - WebP images
        - `.bmp` - Bitmap images
        """)


def render_upload():
    """Render the upload page."""
    st.markdown("## Upload Documents")
    st.markdown("Upload files to add them to your knowledge base.")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=["txt", "pdf", "md", "png", "jpg", "jpeg", "webp", "bmp"],
        accept_multiple_files=True,
        help="You can upload multiple files at once"
    )
    
    # Metadata input
    st.markdown("### Metadata (Optional)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        meta_key = st.text_input("Key", placeholder="e.g., patient_id, category, source")
    
    with col2:
        meta_value = st.text_input("Value", placeholder="e.g., 12345, medical, research")
    
    # Build metadata dict
    metadata = {}
    if meta_key and meta_value:
        metadata[meta_key] = meta_value
    
    # Additional metadata fields
    with st.expander("➕ Add more metadata fields"):
        extra_meta = st.text_area(
            "Additional metadata (JSON format)",
            placeholder='{"author": "John Doe", "date": "2024-01-15"}',
            help="Enter additional metadata as a JSON object"
        )
        
        if extra_meta:
            try:
                extra = json.loads(extra_meta)
                metadata.update(extra)
            except json.JSONDecodeError:
                st.warning("Invalid JSON format for additional metadata")
    
    if metadata:
        st.info(f"Metadata to be added: `{json.dumps(metadata)}`")
    
    # Upload button
    if uploaded_files:
        st.markdown("---")
        st.markdown(f"### Files to Upload ({len(uploaded_files)})")
        
        # Preview files
        for file in uploaded_files:
            file_size = len(file.getvalue()) / 1024  # KB
            st.markdown(f"- **{file.name}** ({file_size:.1f} KB) - {file.type}")
        
        if st.button("Upload All Files", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_container = st.container()
            
            results = []
            
            for i, file in enumerate(uploaded_files):
                with status_container:
                    st.info(f"Uploading: {file.name}...")
                
                try:
                    result = upload_file(file, metadata)
                    results.append({"file": file.name, "result": result})
                except Exception as e:
                    results.append({"file": file.name, "result": {"error": str(e)}})
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            # Show results
            status_container.empty()
            
            st.markdown("### Upload Results")
            
            success_count = 0
            for r in results:
                if "error" not in r["result"]:
                    success_count += 1
                    st.markdown(f"""
                    <div class="success-box">
                        ✅ <strong>{r['file']}</strong> - Uploaded successfully<br>
                        <small>Inserted: {r['result'].get('result', {}).get('inserted', 'N/A')} chunks | 
                        Type: {r['result'].get('result', {}).get('type', 'N/A')}</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="error-box">
                        ❌ <strong>{r['file']}</strong> - Upload failed<br>
                        <small>Error: {r['result'].get('error', 'Unknown error')}</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.success(f"Completed: {success_count}/{len(results)} files uploaded successfully")


def render_query():
    """Render the query page."""
    st.markdown("## Query Your Documents")
    st.markdown("Ask questions and get AI-powered answers with citations.")
    
    # Query input
    query = st.text_area(
        "Enter your question",
        placeholder="What are the main findings in the documents?",
        height=100,
        help="Ask any question about your uploaded documents"
    )
    
    # Query options
    st.markdown("### Query Options")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        top_k = st.slider("Top K Results", 1, 20, 5, help="Number of documents to retrieve")
    
    with col2:
        mode = st.selectbox(
            "Retrieval Mode",
            ["multimodal", "text", "image"],
            help="Choose which types of documents to search"
        )
    
    with col3:
        use_reranker = st.checkbox(
            "Use Reranker",
            value=False,
            help="Apply cross-encoder reranking for better relevance"
        )
    
    with col4:
        use_dag = st.checkbox(
            "Use DAG Pipeline",
            value=False,
            help="Use the DAG-based orchestrator for more structured execution"
        )
    
    # Query button
    if st.button("Search & Generate Answer", type="primary", use_container_width=True, disabled=not query):
        if not query.strip():
            st.warning("Please enter a question")
            return
        
        with st.spinner("Processing your query..."):
            start_time = time.time()
            
            result = query_rag(
                query=query,
                top_k=top_k,
                mode=mode,
                use_reranker=use_reranker,
                use_dag=use_dag
            )
            
            elapsed = time.time() - start_time
        
        if "error" in result:
            st.error(f"Query failed: {result.get('error', 'Unknown error')}")
            return
        
        # Display results
        st.markdown("---")
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Latency", f"{result.get('latency_ms', 0):.0f} ms")
        
        with col2:
            st.metric("Evidence", len(result.get('evidence', [])))
        
        with col3:
            st.metric("Citations", len(result.get('citations', [])))
        
        with col4:
            is_hallucinated = result.get('is_hallucinated', False)
            if is_hallucinated:
                st.metric("⚠️ Status", "Flagged")
            else:
                st.metric("✅ Status", "Verified")
        
        st.markdown("---")
        
        # Answer section
        st.markdown("###Answer")
        
        # Hallucination warning
        if result.get('is_hallucinated'):
            st.markdown("""
            <div class="hallucination-warning">
                ⚠️ <strong>Warning:</strong> This response may contain information not fully grounded in the retrieved evidence. 
                Please verify the claims against the source documents.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="hallucination-ok">
                ✅ Response appears to be well-grounded in the retrieved evidence.
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(result.get('answer', 'No answer generated'))
        
        # Citations
        if result.get('citations'):
            st.markdown("#### 📎 Citations Used")
            citations_html = " ".join([
                f'<span class="citation-tag">[{c}]</span>' 
                for c in result.get('citations', [])
            ])
            st.markdown(citations_html, unsafe_allow_html=True)
        
        # Evidence section
        st.markdown("---")
        st.markdown("### Retrieved Evidence")
        
        evidence = result.get('evidence', [])
        
        if not evidence:
            st.info("No evidence retrieved")
        else:
            for i, ev in enumerate(evidence):
                with st.expander(
                    f"📄 {ev.get('id', f'Evidence {i+1}')} "
                    f"(Score: {ev.get('fused_score', ev.get('score', 0)):.3f})",
                    expanded=(i < 2)  # Expand first 2 by default
                ):
                    modal = ev.get('modal', 'text')
                    metadata = ev.get('metadata', {})
                    
                    # Display metadata
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        st.markdown(f"**Type:** {modal.upper()}")
                        st.markdown(f"**Source:** {metadata.get('source', 'N/A')}")
                        if 'chunk_index' in metadata:
                            st.markdown(f"**Chunk:** {metadata.get('chunk_index')}")
                    
                    with col2:
                        if modal == 'text':
                            st.markdown("**Content:**")
                            st.text_area(
                                "Document content",
                                ev.get('document', '')[:1000],
                                height=150,
                                disabled=True,
                                label_visibility="collapsed",
                                key=f"evidence_{i}"
                            )
                        else:
                            st.markdown("**Image:**")
                            image_path = ev.get('document', '')
                            caption = metadata.get('caption', 'No caption')
                            st.markdown(f"Path: `{image_path}`")
                            st.markdown(f"Caption: {caption}")
                            
                            # Try to display image if it exists
                            if os.path.exists(image_path):
                                try:
                                    st.image(image_path, caption=caption, width=300)
                                except Exception:
                                    st.warning("Could not display image")


def render_statistics():
    """Render the statistics page."""
    st.markdown("##System Statistics")
    
    # Refresh button
    if st.button("🔄 Refresh Stats", use_container_width=False):
        st.rerun()
    
    # Get stats
    stats = get_stats()
    
    if "error" in stats:
        st.error(f"Failed to get statistics: {stats.get('error')}")
        return
    
    if "stats" not in stats:
        st.warning("No statistics available")
        return
    
    stats_data = stats["stats"]
    
    # Collection stats
    st.markdown("### Collection Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        text_stats = stats_data.get("text_collection", {})
        st.markdown("""
        <div class="metric-card">
            <h2>📄 Text Collection</h2>
            <h1 style="color: #1E88E5;">{}</h1>
            <p>documents indexed</p>
            <small>Collection: {}</small>
        </div>
        """.format(
            text_stats.get("count", 0),
            text_stats.get("name", "N/A")
        ), unsafe_allow_html=True)
    
    with col2:
        image_stats = stats_data.get("image_collection", {})
        st.markdown("""
        <div class="metric-card">
            <h2>🖼️ Image Collection</h2>
            <h1 style="color: #43A047;">{}</h1>
            <p>images indexed</p>
            <small>Collection: {}</small>
        </div>
        """.format(
            image_stats.get("count", 0),
            image_stats.get("name", "N/A")
        ), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # API Info
    st.markdown("### 🔧 API Information")
    
    api_info = get_api_info()
    
    if api_info:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"**Service:** {api_info.get('service', 'N/A')}")
        
        with col2:
            st.markdown(f"**Version:** {api_info.get('version', 'N/A')}")
        
        with col3:
            st.markdown(f"**LLM Backend:** {api_info.get('llm_backend', 'N/A')}")
        
        st.markdown(f"**Model:** `{api_info.get('model', 'N/A')}`")
    
    # Visualization
    st.markdown("---")
    st.markdown("### Collection Distribution")
    
    import plotly.graph_objects as go
    
    text_count = stats_data.get("text_collection", {}).get("count", 0)
    image_count = stats_data.get("image_collection", {}).get("count", 0)
    
    if text_count > 0 or image_count > 0:
        fig = go.Figure(data=[
            go.Pie(
                labels=['Text Documents', 'Images'],
                values=[text_count, image_count],
                hole=0.4,
                marker_colors=['#1E88E5', '#43A047']
            )
        ])
        
        fig.update_layout(
            title="Document Distribution",
            showlegend=True,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No documents indexed yet. Upload some files to see statistics.")


def render_settings():
    """Render the settings page."""
    st.markdown("##Settings")
    
    # API Configuration
    st.markdown("###API Configuration")
    
    new_api_url = st.text_input(
        "API Base URL",
        value=API_BASE_URL,
        help="The base URL of the RAG API"
    )
    
    if new_api_url != API_BASE_URL:
        st.warning("Changes to API URL require restarting the Streamlit app")
    
    # Cache Management
    st.markdown("---")
    st.markdown("### Cache Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clear Query Cache", type="secondary", use_container_width=True):
            result = clear_cache()
            if "error" in result:
                st.error(f"Failed to clear cache: {result.get('error')}")
            else:
                st.success(f"Cache cleared! {result.get('cleared', 0)} entries removed.")
    
    with col2:
        if st.button("Refresh App", type="secondary", use_container_width=True):
            st.rerun()
    
    # About
    st.markdown("---")
    st.markdown("### About")
    
    st.markdown("""
    **Multimodal RAG System** is a retrieval-augmented generation system that:
    
    - Indexes text documents, PDFs, and images
    - Uses hybrid retrieval (text + image embeddings)
    - Generates grounded answers with citations
    - Detects potential hallucinations
    
    **Technologies:**
    - FastAPI backend
    - ChromaDB for vector storage
    - Sentence Transformers for text embeddings
    - CLIP for image embeddings
    - LLM integration (Groq/OpenAI)
    """)
    
    # Health Check Details
    st.markdown("---")
    st.markdown("### Health Check")
    
    if st.button("Run Health Check"):
        checks = {
            "API Health": check_api_health(),
            "Stats Endpoint": "stats" in get_stats(),
            "API Info": get_api_info() is not None
        }
        
        for check, status in checks.items():
            if status:
                st.success(f"✅ {check}: OK")
            else:
                st.error(f"❌ {check}: Failed")


# ---- Main Application ----

def main():
    """Main application entry point."""
    # Render sidebar and get selected page
    page = render_sidebar()
    
    # Route to appropriate page
    if page == "🏠 Home":
        render_home()
    elif page == "📤 Upload Documents":
        render_upload()
    elif page == "💬 Query":
        render_query()
    elif page == "📊 Statistics":
        render_statistics()
    elif page == "⚙️ Settings":
        render_settings()


if __name__ == "__main__":
    main()