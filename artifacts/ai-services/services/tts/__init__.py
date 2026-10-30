"""
TTS Package

Text-to-Speech service using Kokoro TTS.

Provides:
- Audio synthesis from text
- Text segmentation for large inputs
- Caching by text+voice
- Audio file storage and retrieval
- Integration with transformation engine
"""

from .service import TTSService, tts_service
from .types import (
    TTSRequest,
    TTSResponse,
    TTSConfig,
    TTSVoiceInfo,
    TTSVoice,
    AudioFileInfo,
    VOICE_CATALOG,
    DEFAULT_VOICE,
    MAX_TEXT_LENGTH,
    SEGMENT_MAX_LENGTH,
    get_voice_info,
    get_default_voices,
)

__all__ = [
    "TTSService",
    "tts_service",
    "TTSRequest",
    "TTSResponse",
    "TTSConfig",
    "TTSVoiceInfo",
    "TTSVoice",
    "AudioFileInfo",
    "VOICE_CATALOG",
    "DEFAULT_VOICE",
    "MAX_TEXT_LENGTH",
    "SEGMENT_MAX_LENGTH",
    "get_voice_info",
    "get_default_voices",
]