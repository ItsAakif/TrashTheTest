"""
TTS API Routes

API endpoints for Text-to-Speech:
- POST /api/tts/generate - Generate audio from text
- GET /api/tts/audio/{id} - Retrieve audio file
- GET /api/tts/project/{id} - List project audio
- DELETE /api/tts/audio/{id} - Delete audio
- GET /api/tts/voices - List available voices
- GET /api/tts/health - Health check
"""

import uuid
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from database.engine import get_db
from .service import tts_service
from .types import (
    TTSRequest,
    TTSResponse,
    TTSListResponse,
    TTSHealthResponse,
    TTSVoiceInfo,
    MAX_TEXT_LENGTH,
    SEGMENT_MAX_LENGTH,
)

router = APIRouter(prefix="/api/tts", tags=["tts"])


# ==================== Request Models ====================

class GenerateTTSRequest(BaseModel):
    """Request to generate TTS audio"""
    text: str
    voice: str = "af_sarah"
    speed: float = 1.0
    project_id: Optional[uuid.UUID] = None
    content_type: Optional[str] = None
    content_id: Optional[uuid.UUID] = None


# ==================== Endpoints ====================

@router.post("/generate", response_model=TTSResponse)
async def generate_audio(
    request: GenerateTTSRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Generate audio from text.
    
    The service will:
    - Cache audio for identical requests
    - Segment large text into manageable chunks
    - Store audio files for retrieval
    
    Request body:
    - text: Text to convert to speech (max 50000 chars)
    - voice: Voice identifier (default: af_sarah)
    - speed: Playback speed 0.5-2.0 (default: 1.0)
    - project_id: Optional project association
    - content_type: Optional content type (notes, quiz, etc.)
    - content_id: Optional content ID
    """
    try:
        result = await tts_service.synthesize(
            text=request.text,
            voice=request.voice,
            speed=request.speed,
            project_id=request.project_id,
            content_type=request.content_type,
            content_id=request.content_id,
            db=db
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")


@router.get("/audio/{audio_id}", response_model=TTSResponse)
async def get_audio(
    audio_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Get audio file by ID.
    
    Returns audio metadata and file path.
    """
    audio = await tts_service.get_audio(audio_id, db)
    
    if not audio:
        raise HTTPException(status_code=404, detail="Audio not found")
    
    return audio


@router.get("/audio/{audio_id}/stream")
async def stream_audio(
    audio_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Stream audio file.
    
    Returns the audio file for playback.
    """
    audio = await tts_service.get_audio(audio_id, db)
    
    if not audio:
        raise HTTPException(status_code=404, detail="Audio not found")
    
    if not audio.file_path:
        raise HTTPException(status_code=404, detail="Audio file not available")
    
    return FileResponse(
        audio.file_path,
        media_type=f"audio/{audio.format}",
        headers={"Content-Disposition": f"inline; filename={audio_id}.{audio.format}"}
    )


@router.get("/project/{project_id}", response_model=TTSListResponse)
async def list_project_audio(
    project_id: uuid.UUID,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    db: AsyncSession = Depends(get_db)
):
    """
    List all TTS audio for a project.
    
    Returns paginated list of audio files.
    """
    offset = (page - 1) * page_size
    
    audio_files = await tts_service.list_project_audio(
        project_id=project_id,
        db=db,
        limit=page_size,
        offset=offset
    )
    
    return TTSListResponse(
        audio_files=audio_files,
        total=len(audio_files),  # Simplified - could count properly
        page=page,
        page_size=page_size
    )


@router.delete("/audio/{audio_id}")
async def delete_audio(
    audio_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete audio file and metadata.
    """
    success = await tts_service.delete_audio(audio_id, db)
    
    if not success:
        raise HTTPException(status_code=404, detail="Audio not found")
    
    return {"success": True, "message": f"Audio {audio_id} deleted"}


@router.get("/voices")
async def list_voices():
    """
    Get list of available TTS voices.
    
    Returns voice IDs and descriptions.
    """
    voices = await tts_service.get_voices()
    
    return {
        "voices": [v.model_dump() for v in voices],
        "default": "af_sarah"
    }


@router.get("/health", response_model=TTSHealthResponse)
async def health_check():
    """
    Check TTS service health.
    """
    health = await tts_service.health_check()
    
    return TTSHealthResponse(
        status=health.get("status", "unknown"),
        available_voices=await tts_service.get_voices(),
        default_voice="af_sarah",
        provider=health.get("provider", "unknown")
    )


@router.get("/info")
async def get_tts_info():
    """
    Get TTS service information.
    """
    return {
        "max_text_length": MAX_TEXT_LENGTH,
        "segment_max_length": SEGMENT_MAX_LENGTH,
        "supported_formats": ["mp3", "wav", "ogg"],
        "speed_range": {"min": 0.5, "max": 2.0, "default": 1.0}
    }


__all__ = ["router"]