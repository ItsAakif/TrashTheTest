"""
TTS Service Types

Type definitions for the Text-to-Speech service.
"""

import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class TTSVoice(str, Enum):
    """Available TTS voices"""
    # American English - Female
    AF_SARAH = "af_sarah"
    AF_JESSICA = "af_jessica"
    AF_NOVA = "af_nova"
    # American English - Male
    AM_ADAM = "am_adam"
    AM_MICHAEL = "am_michael"
    # British English - Female
    BF_EMMA = "bf_emma"
    BF_ISABELLA = "bf_isabella"
    # British English - Male
    BM_DANIEL = "bm_daniel"
    BM_GEORGE = "bm_george"


class TTSVoiceInfo(BaseModel):
    """Information about a TTS voice"""
    voice_id: str
    description: str
    language: str
    gender: str


class TTSConfig(BaseModel):
    """TTS configuration"""
    voice: str = "af_sarah"
    speed: float = 1.0  # 0.5 to 2.0
    pitch: float = 1.0  # 0.5 to 2.0
    volume: float = 1.0  # 0.0 to 1.0
    output_format: str = "mp3"  # mp3, wav, ogg


class TTSRequest(BaseModel):
    """Request to generate audio"""
    text: str = Field(..., min_length=1, description="Text to convert to speech")
    voice: str = Field(default="af_sarah", description="Voice identifier")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Playback speed")
    project_id: Optional[uuid.UUID] = Field(default=None, description="Associated project")
    content_type: Optional[str] = Field(default=None, description="Type of content (notes, quiz, etc.)")
    content_id: Optional[uuid.UUID] = Field(default=None, description="ID of source content")


class TTSResponse(BaseModel):
    """Response for TTS generation"""
    audio_id: uuid.UUID
    project_id: Optional[uuid.UUID]
    text_preview: str  # First 100 chars
    duration_seconds: float
    voice: str
    format: str
    file_path: Optional[str] = None
    file_url: Optional[str] = None
    created_at: datetime
    cached: bool = False


class TTSListResponse(BaseModel):
    """Response for listing TTS audio"""
    audio_files: List[TTSResponse]
    total: int
    page: int = 1
    page_size: int = 20


class AudioFileInfo(BaseModel):
    """Audio file metadata"""
    id: uuid.UUID
    project_id: Optional[uuid.UUID]
    file_path: str
    file_name: str
    format: str
    size_bytes: int
    duration_seconds: float
    voice: str
    text_hash: str  # Hash of source text for caching
    content_type: Optional[str] = None
    content_id: Optional[uuid.UUID] = None
    created_at: datetime
    accessed_at: Optional[datetime] = None


class TTSHealthResponse(BaseModel):
    """TTS service health status"""
    status: str
    available_voices: List[TTSVoiceInfo]
    default_voice: str
    provider: str


# Voice descriptions for the system
VOICE_CATALOG = {
    "af_sarah": TTSVoiceInfo(
        voice_id="af_sarah",
        description="American English, female, bright and energetic",
        language="en-US",
        gender="female"
    ),
    "af_jessica": TTSVoiceInfo(
        voice_id="af_jessica",
        description="American English, female, calm and professional",
        language="en-US",
        gender="female"
    ),
    "af_nova": TTSVoiceInfo(
        voice_id="af_nova",
        description="American English, female, expressive and dynamic",
        language="en-US",
        gender="female"
    ),
    "am_adam": TTSVoiceInfo(
        voice_id="am_adam",
        description="American English, male, calm and trustworthy",
        language="en-US",
        gender="male"
    ),
    "am_michael": TTSVoiceInfo(
        voice_id="am_michael",
        description="American English, male, expressive and engaging",
        language="en-US",
        gender="male"
    ),
    "bf_emma": TTSVoiceInfo(
        voice_id="bf_emma",
        description="British English, female, bright and professional",
        language="en-GB",
        gender="female"
    ),
    "bf_isabella": TTSVoiceInfo(
        voice_id="bf_isabella",
        description="British English, female, calm and elegant",
        language="en-GB",
        gender="female"
    ),
    "bm_daniel": TTSVoiceInfo(
        voice_id="bm_daniel",
        description="British English, male, calm and authoritative",
        language="en-GB",
        gender="male"
    ),
    "bm_george": TTSVoiceInfo(
        voice_id="bm_george",
        description="British English, male, bright and friendly",
        language="en-GB",
        gender="male"
    ),
}


# Constants
MAX_TEXT_LENGTH = 50000  # Max characters before segmentation
SEGMENT_MAX_LENGTH = 2000  # Max characters per segment
DEFAULT_VOICE = "af_sarah"
AUDIO_STORAGE_PATH = "audio/tts"


def get_voice_info(voice_id: str) -> Optional[TTSVoiceInfo]:
    """Get voice information by ID"""
    return VOICE_CATALOG.get(voice_id)


def get_default_voices() -> List[str]:
    """Get list of default voice IDs"""
    return list(VOICE_CATALOG.keys())