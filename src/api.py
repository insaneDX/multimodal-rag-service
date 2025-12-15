from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
import json
import tempfile
import os
from pathlib import Path

import src.logger_config
from loguru import logger

from src.ingestion import upload_and_ingest, get_collection_stats
from prometheus_client import generate_latest, REGISTRY


router = APIRouter(prefix="/api/v1", tags=["ingestion"])

ALLOWED_EXTENSIONS = {'.txt', '.pdf', '.png', '.jpeg', '.jpg', '.md', '.webp', '.bmp'}


@router.post("/upload")
async def upload(
    file: UploadFile = File(...), 
    metadata: str = Form("{}")
):
    """
    Upload and ingest a single file (text, PDF, or image).

    Args:
        file: The file to upload
        metadata: Optional JSON string for additional metadata (e.g., {"patient_id": "123"})

    Returns:
        Ingestion result with status and details
    """
    # Check if filename exists
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    # Get the file extension (e.g., .txt, .pdf, etc.)
    file_suffix = Path(file.filename).suffix.lower()

    # Check if file type is allowed
    if file_suffix not in ALLOWED_EXTENSIONS:
        logger.warning(f"Attempted upload of unsupported file type: {file.filename}")
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file_suffix}. Allowed: {ALLOWED_EXTENSIONS}"
        )

    # Parse metadata JSON
    try:
        meta = json.loads(metadata)
        if not isinstance(meta, dict):
            raise ValueError("Metadata must be a JSON object")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid metadata JSON: {metadata} -- Error: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON in 'metadata' field.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Save the file to a temporary file
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            temp_path = tmp_file.name
        
        logger.info(f"Processing uploaded file: {file.filename} -> saved to temp path: {temp_path}")
        # Inject original filename into metadata
        #final_metadata = {**meta, "filename": file.filename}
        # Perform ingestion
        try:
            res = upload_and_ingest(
                temp_path, 
                metadata=meta,
                original_filename=file.filename
            )

        except ValueError as e:
            logger.error(f"Ingestion error for file {file.filename}: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.exception(f"Error during ingestion of file {file.filename}")
            raise HTTPException(
                status_code=500, 
                detail="Ingestion failed. Check server logs for details."
            )
        
        return {
            "status": "ok", 
            "filename": file.filename,
            "result": res
        }

    finally:
        # Cleanup temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.warning(f"Failed to remove temp file {temp_path}: {e}")


@router.get("/stats")
async def get_stats():
    """Get statistics about the ingested documents."""
    try:
        stats = get_collection_stats()
        return {"status": "ok", "stats": stats}
    except Exception as e:
        logger.exception("Failed to get collection stats")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@router.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        generate_latest(REGISTRY), 
        media_type='text/plain; version=0.0.4; charset=utf-8'
    )