"""
TTS Service

Text-to-Speech service using Kokoro TTS provider.
Handles text segmentation, caching, and audio file storage.
"""

import uuid
import hashlib
import os
import io
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession

from factory import get_tts
from database import repositories
from services.cache import get_cache_service, CacheType

from .types import (
    TTSRequest,
    TTSResponse,
    TTSConfig,
    TTSVoiceInfo,
    AudioFileInfo,
    VOICE_CATALOG,
    DEFAULT_VOICE,
    MAX_TEXT_LENGTH,
    SEGMENT_MAX_LENGTH,
    AUDIO_STORAGE_PATH,
)

logger = logging.getLogger(__name__)


class TTSService:
    """
    Text-to-Speech service using Kokoro TTS.
    
    Features:
    - Text segmentation for large inputs
    - Caching by text+voice hash
    - Audio file storage
    - Integration with generation engine
    """
    
    def __init__(self):
        self._tts_provider = None
        self._cache = None
    
    async def _get_provider(self):
        """Get TTS provider (lazy loading)"""
        if self._tts_provider is None:
            self._tts_provider = get_tts()
        return self._tts_provider
    
    async def _get_cache(self):
        """Get cache service (lazy loading)"""
        if self._cache is None:
            self._cache = get_cache_service()
        return self._cache
    
    def _compute_text_hash(self, text: str, voice: str, speed: float = 1.0) -> str:
        """Compute hash for cache key"""
        key_material = f"{text}|{voice}|{speed}"
        return hashlib.sha256(key_material.encode("utf-8")).hexdigest()[:24]
    
    def _generate_cache_key(self, text_hash: str, voice: str) -> str:
        """Generate cache key for TTS"""
        return f"tts:{voice}:{text_hash}"
    
    def _segment_text(self, text: str) -> List[str]:
        """
        Segment text into manageable chunks.
        
        Splits on sentence boundaries when possible.
        """
        if len(text) <= SEGMENT_MAX_LENGTH:
            return [text]
        
        segments = []
        current_segment = ""
        
        # Split by common sentence terminators
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            if len(current_segment) + len(sentence) <= SEGMENT_MAX_LENGTH:
                current_segment += " " + sentence if current_segment else sentence
            else:
                if current_segment:
                    segments.append(current_segment.strip())
                # If single sentence is too long, force split
                if len(sentence) > SEGMENT_MAX_LENGTH:
                    # Split by words
                    words = sentence.split()
                    temp = ""
                    for word in words:
                        if len(temp) + len(word) + 1 <= SEGMENT_MAX_LENGTH:
                            temp += " " + word if temp else word
                        else:
                            if temp:
                                segments.append(temp)
                            temp = word
                    current_segment = temp
                else:
                    current_segment = sentence
        
        if current_segment:
            segments.append(current_segment.strip())
        
        return segments
    
    async def synthesize(
        self,
        text: str,
        voice: str = DEFAULT_VOICE,
        speed: float = 1.0,
        project_id: Optional[uuid.UUID] = None,
        content_type: Optional[str] = None,
        content_id: Optional[uuid.UUID] = None,
        db: Optional[AsyncSession] = None
    ) -> TTSResponse:
        """
        Synthesize text to speech.
        
        Handles:
        - Caching of existing audio
        - Text segmentation for large inputs
        - Audio file storage
        
        Args:
            text: Text to convert to speech
            voice: Voice identifier
            speed: Playback speed (0.5-2.0)
            project_id: Associated project
            content_type: Type of content (notes, quiz, etc.)
            content_id: ID of source content
            db: Database session
            
        Returns:
            TTSResponse with audio information
        """
        # Validate input
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        text = text.strip()
        
        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(
                f"Text too long ({len(text)} chars). "
                f"Maximum is {MAX_TEXT_LENGTH} characters."
            )
        
        # Check cache
        cache = await self._get_cache()
        text_hash = self._compute_text_hash(text, voice, speed)
        cache_key = self._generate_cache_key(text_hash, voice)
        
        cached_audio = await cache.get(cache_key, cache_type=CacheType.USER_SESSION)
        if cached_audio:
            logger.info(f"TTS cache hit: {cache_key}")
            
            # Return cached result
            return TTSResponse(
                audio_id=uuid.UUID(cached_audio["audio_id"]),
                project_id=project_id,
                text_preview=text[:100],
                duration_seconds=cached_audio["duration"],
                voice=voice,
                format=cached_audio.get("format", "mp3"),
                file_path=cached_audio.get("file_path"),
                file_url=cached_audio.get("file_url"),
                created_at=datetime.fromisoformat(cached_audio["created_at"]),
                cached=True
            )
        
        # Get TTS provider
        provider = await self._get_provider()
        
        # Segment text if needed
        segments = self._segment_text(text)
        
        audio_segments = []
        total_duration = 0.0
        
        for i, segment in enumerate(segments):
            logger.debug(f"Synthesizing segment {i+1}/{len(segments)}")
            
            try:
                result = await provider.synthesize(segment, voice)
                
                if result.audio_data:
                    audio_segments.append(result.audio_data)
                    total_duration += result.duration_ms / 1000.0
                elif result.audio_path and os.path.exists(result.audio_path):
                    with open(result.audio_path, "rb") as f:
                        audio_segments.append(f.read())
                    total_duration += result.duration_ms / 1000.0
                    
            except Exception as e:
                logger.error(f"Failed to synthesize segment {i+1}: {e}")
                # Continue with other segments
                continue
        
        # Combine audio segments
        if not audio_segments:
            raise ValueError("Failed to generate audio for any segment")
        
        combined_audio = b"".join(audio_segments)
        
        # Save audio file
        audio_id = uuid.uuid4()
        file_path = await self._save_audio_file(
            audio_id=audio_id,
            audio_data=combined_audio,
            project_id=project_id
        )
        
        # Store in database
        if db and project_id:
            await self._store_audio_metadata(
                db=db,
                audio_id=audio_id,
                project_id=project_id,
                file_path=file_path,
                text_hash=text_hash,
                voice=voice,
                duration=total_duration,
                content_type=content_type,
                content_id=content_id
            )
        
        # Cache the result
        cache_data = {
            "audio_id": str(audio_id),
            "duration": total_duration,
            "format": "mp3",
            "file_path": file_path,
            "created_at": datetime.utcnow().isoformat()
        }
        await cache.set(cache_key, cache_data, cache_type=CacheType.USER_SESSION)
        
        logger.info(f"TTS generated: {audio_id} ({total_duration:.2f}s)")
        
        return TTSResponse(
            audio_id=audio_id,
            project_id=project_id,
            text_preview=text[:100],
            duration_seconds=total_duration,
            voice=voice,
            format="mp3",
            file_path=file_path,
            created_at=datetime.utcnow(),
            cached=False
        )
    
    async def _save_audio_file(
        self,
        audio_id: uuid.UUID,
        audio_data: bytes,
        project_id: Optional[uuid.UUID]
    ) -> str:
        """Save audio data to file"""
        # Create directory structure
        if project_id:
            base_path = os.path.join(AUDIO_STORAGE_PATH, str(project_id))
        else:
            base_path = os.path.join(AUDIO_STORAGE_PATH, "general")
        
        os.makedirs(base_path, exist_ok=True)
        
        # Save file
        file_name = f"{audio_id}.mp3"
        file_path = os.path.join(base_path, file_name)
        
        with open(file_path, "wb") as f:
            f.write(audio_data)
        
        logger.debug(f"Saved audio file: {file_path}")
        return file_path
    
    async def _store_audio_metadata(
        self,
        db: AsyncSession,
        audio_id: uuid.UUID,
        project_id: uuid.UUID,
        file_path: str,
        text_hash: str,
        voice: str,
        duration: float,
        content_type: Optional[str] = None,
        content_id: Optional[uuid.UUID] = None
    ):
        """Store audio metadata in database"""
        # For now, we'll store this in the generated content cache
        # In production, you'd have a dedicated audio_files table
        from database.models import GeneratedContentCache
        
        cache_entry = GeneratedContentCache(
            project_id=project_id,
            content_type="tts_audio",
            content_hash=text_hash,
            content_data={
                "audio_id": str(audio_id),
                "file_path": file_path,
                "voice": voice,
                "duration": duration,
                "content_type": content_type,
                "content_id": str(content_id) if content_id else None
            }
        )
        
        db.add(cache_entry)
        await db.commit()
    
    async def get_audio(
        self,
        audio_id: uuid.UUID,
        db: AsyncSession
    ) -> Optional[TTSResponse]:
        """Retrieve audio by ID"""
        from sqlalchemy import select
        from database.models import GeneratedContentCache
        
        result = await db.execute(
            select(GeneratedContentCache).where(
                GeneratedContentCache.content_type == "tts_audio",
                GeneratedContentCache.content_data["audio_id"].astext == str(audio_id)
            )
        )
        
        cache_entry = result.scalars().first()
        
        if not cache_entry:
            return None
        
        data = cache_entry.content_data
        
        return TTSResponse(
            audio_id=audio_id,
            project_id=cache_entry.project_id,
            text_preview="",  # Not stored
            duration_seconds=data.get("duration", 0),
            voice=data.get("voice", DEFAULT_VOICE),
            format="mp3",
            file_path=data.get("file_path"),
            created_at=cache_entry.created_at,
            cached=True
        )
    
    async def list_project_audio(
        self,
        project_id: uuid.UUID,
        db: AsyncSession,
        limit: int = 20,
        offset: int = 0
    ) -> List[TTSResponse]:
        """List all TTS audio for a project"""
        from sqlalchemy import select
        from database.models import GeneratedContentCache
        
        result = await db.execute(
            select(GeneratedContentCache)
            .where(
                GeneratedContentCache.project_id == project_id,
                GeneratedContentCache.content_type == "tts_audio"
            )
            .order_by(GeneratedContentCache.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        
        audio_files = []
        
        for entry in result.scalars().all():
            data = entry.content_data
            audio_id = uuid.UUID(data["audio_id"])
            
            audio_files.append(TTSResponse(
                audio_id=audio_id,
                project_id=project_id,
                text_preview="",  # Not stored
                duration_seconds=data.get("duration", 0),
                voice=data.get("voice", DEFAULT_VOICE),
                format="mp3",
                file_path=data.get("file_path"),
                created_at=entry.created_at,
                cached=True
            ))
        
        return audio_files
    
    async def delete_audio(
        self,
        audio_id: uuid.UUID,
        db: AsyncSession
    ) -> bool:
        """Delete audio file and metadata"""
        # Get the audio first
        audio = await self.get_audio(audio_id, db)
        
        if not audio:
            return False
        
        # Delete file
        if audio.file_path and os.path.exists(audio.file_path):
            os.remove(audio.file_path)
        
        # Delete database entry
        from sqlalchemy import delete
        from database.models import GeneratedContentCache
        
        await db.execute(
            delete(GeneratedContentCache).where(
                GeneratedContentCache.content_data["audio_id"].astext == str(audio_id)
            )
        )
        await db.commit()
        
        # Invalidate cache
        cache = await self._get_cache()
        # Can't easily invalidate without text hash, so leave it to TTL
        
        return True
    
    async def get_voices(self) -> List[TTSVoiceInfo]:
        """Get list of available voices"""
        return list(VOICE_CATALOG.values())
    
    async def health_check(self) -> Dict[str, Any]:
        """Check TTS service health"""
        try:
            provider = await self._get_provider()
            
            return {
                "status": "healthy",
                "provider": "kokoro",
                "available_voices": len(VOICE_CATALOG),
                "default_voice": DEFAULT_VOICE
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# Global service instance
tts_service = TTSService()


__all__ = [
    "TTSService",
    "tts_service",
]