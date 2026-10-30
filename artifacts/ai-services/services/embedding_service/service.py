"""
Embedding Generation Service

Service for generating embeddings for document chunks using local models.
Supports batch processing, caching, and incremental generation.
"""

import asyncio
import logging
import uuid
import time
import struct
from typing import Optional, List, Dict, Any
from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from factory import get_embedding as _factory_embedding
from .types import (
    EmbeddingChunkInput,
    EmbeddingBatchInput,
    EmbeddingProgress,
    EmbeddingResult,
    EmbeddingJobInput,
    EmbeddingConfig,
)
from database.models import Chunk, JobStatus
from database import repositories

logger = logging.getLogger(__name__)


class LocalEmbeddingService:
    """
    Service for generating embeddings using local models.
    
    Features:
    - Batch processing for efficiency
    - Content hash-based caching
    - Incremental embedding (only new chunks)
    - Graceful error handling with retry
    - Progress tracking
    
    Uses BGE-small (384 dimensions) as the default model.
    """
    
    def __init__(
        self,
        embedding_provider=None,
        config: Optional[EmbeddingConfig] = None
    ):
        self.provider = embedding_provider or _factory_embedding()
        self.config = config or EmbeddingConfig()

        # Embedding cache keyed by (embedding_model, content_hash) layered on
        # the existing CacheService when available, else this memory dict.
        self._embedding_cache: Dict[str, bytes] = {}
        try:
            from services.cache import get_cache_service as _cache

            self._cache_backend = _cache()
        except Exception:
            self._cache_backend = None
    
    def _cache_key(self, model: str, content_hash: str) -> str:
        return f"emb:{model}:{content_hash}"

    def _cache_get(self, model: str, content_hash: str) -> Optional[bytes]:
        key = self._cache_key(model, content_hash)
        hit = self._embedding_cache.get(key)
        if hit is not None:
            return hit
        backend = getattr(self, "_cache_backend", None)
        if backend is not None:
            try:
                val = backend.get(key) if not asyncio.iscoroutinefunction(backend.get) else None
                if isinstance(val, (bytes, bytearray)):
                    self._embedding_cache[key] = bytes(val)
                    return bytes(val)
            except Exception:
                pass
        return None

    def _cache_set(self, model: str, content_hash: str, value: bytes) -> None:
        key = self._cache_key(model, content_hash)
        self._embedding_cache[key] = value
        backend = getattr(self, "_cache_backend", None)
        if backend is not None:
            try:
                setter = getattr(backend, "set", None)
                if setter is not None and not getattr(setter, "_is_coroutine", False):
                    try:
                        setter(key, value)
                    except TypeError:
                        pass
            except Exception:
                pass

    def _model_name(self) -> str:
        provider = getattr(self, "provider", None)
        name = getattr(provider, "model_name", None)
        if isinstance(name, str) and name:
            return name
        return getattr(self.config, "model_name", None) or "Qwen/Qwen3-Embedding-0.6B"

    def _model_dim(self) -> int:
        provider = getattr(self, "provider", None)
        dim = getattr(provider, "dimensions", None)
        try:
            return int(dim) if dim else int(self.config.model_dimensions or 1024)
        except (TypeError, ValueError):
            return 1024

    def _pack_vector(self, vec: List[float]) -> bytes:
        """Pack one vector's floats to bytes (T403: never index the bytes)."""
        vals = [float(v) for v in vec]
        return struct.pack(f"{len(vals)}f", *vals)

    def _embeddings_to_bytes(self, embeddings: List[List[float]]) -> bytes:
        """Convert float embeddings to bytes (compat wrapper; prefer _pack_vector)."""
        return b"".join(self._pack_vector(emb) for emb in embeddings)

    @staticmethod
    def unpack_vector(blob: bytes) -> List[float]:
        """Round-trip helper: bytes back to float components."""
        data = bytes(blob)
        n = len(data) // 4
        return list(struct.unpack(f"{n}f", data))
    
    async def _load_cached_embeddings(
        self,
        db: AsyncSession,
        chunk_ids: List[uuid.UUID]
    ) -> Dict[str, bytes]:
        """
        Load existing embeddings from database for caching.
        
        Returns a dict of chunk_id -> embedding_bytes
        """
        result = await db.execute(
            select(Chunk).where(
                Chunk.id.in_(chunk_ids),
                Chunk.embedding.isnot(None)
            )
        )
        chunks = result.scalars().all()
        
        cached = {}
        for chunk in chunks:
            if chunk.embedding:
                cached[str(chunk.id)] = chunk.embedding
        
        return cached
    
    async def _check_cache(
        self,
        chunks: List[EmbeddingChunkInput],
        db: AsyncSession
    ) -> tuple[List[EmbeddingChunkInput], List[EmbeddingChunkInput]]:
        """
        Check which chunks have cached embeddings.
        
        Returns: (cached_chunks, uncached_chunks)
        """
        if not self.config.cache_enabled:
            return [], chunks
        
        # Get chunk IDs
        chunk_ids = [uuid.UUID(c.chunk_id) for c in chunks]
        
        # Load existing embeddings from database
        cached_embeddings = await self._load_cached_embeddings(db, chunk_ids)
        
        cached = []
        uncached = []
        
        for chunk in chunks:
            if chunk.chunk_id in cached_embeddings:
                # Use cached embedding - store in memory cache too (same key
                # everywhere: emb:{model}:{hash}, T403).
                key = self._cache_key(self._model_name(), chunk.content_hash)
                self._embedding_cache[key] = cached_embeddings[chunk.chunk_id]
                cached.append(chunk)
            else:
                uncached.append(chunk)
        
        return cached, uncached
    
    async def _generate_embeddings_batch(
        self,
        chunks: List[EmbeddingChunkInput]
    ) -> Dict[str, bytes]:
        """
        Generate embeddings for a batch of chunks.
        
        Returns: dict of content_hash -> embedding_bytes
        """
        # Extract text content
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings using the provider
        result = await self.provider.generate_embeddings(
            texts=texts,
            batch_size=self.config.batch_size
        )
        
        # Convert to bytes and create mapping (T403: pack the actual
        # vector per chunk; the old [0] indexed into bytes yielding an int).
        # Keyed by chunk_id: _update_chunks_with_embeddings resolves rows by
        # id (keying by content_hash left UUID() parsing to fail silently).
        model = self._model_name()
        embeddings_dict = {}
        for i, chunk in enumerate(chunks):
            vec = [float(v) for v in result.embeddings[i]]
            embedding_bytes = self._pack_vector(vec)
            embeddings_dict[chunk.chunk_id] = embedding_bytes
            # Also cache in memory (identical key everywhere)
            self._embedding_cache[self._cache_key(model, chunk.content_hash)] = embedding_bytes

        return embeddings_dict

    async def _update_chunks_with_embeddings(
        self,
        db: AsyncSession,
        chunk_embeddings: Dict[str, bytes]
    ):
        """Update chunks with their embeddings in the database.

        Writes the packed vector plus embedding_model/embedding_dim tags so a
        stored row round-trips to the configured dimension. Flush only: the
        worker owns the commit (T405 scope rule).
        """
        model = self._model_name()
        dim = self._model_dim()
        for chunk_id_str, embedding_bytes in chunk_embeddings.items():
            chunk_id = uuid.UUID(chunk_id_str)
            await db.execute(
                update(Chunk)
                .where(Chunk.id == chunk_id)
                .values(embedding=embedding_bytes,
                        embedding_model=model,
                        embedding_dim=dim)
            )

        await db.flush()
    
    async def _update_job_progress(
        self,
        db: AsyncSession,
        job_id: uuid.UUID,
        progress: int,
        message: str
    ):
        """Update job progress without clobbering worker-owned output_data.

        T404/T409: update_job_status merges keys, so _stages_done survives.
        Scaled into the worker's embed slice (80-95) so the job percentage
        stays monotonic with the surrounding stage reports.
        """
        pct = 80 + int(max(0, min(100, int(progress))) * 15 / 100)
        await repositories.update_job_status(
            db=db,
            job_id=job_id,
            status=JobStatus.PROCESSING,
            progress=pct,
            output_data={"message": message}
        )
        # Commit so pollers see per-batch movement (same long-transaction
        # visibility fix as the worker stage machine). Best-effort: the
        # worker owns the final commit.
        try:
            await db.commit()
        except Exception:
            pass
    
    async def generate_embeddings(
        self,
        document_id: uuid.UUID,
        project_id: uuid.UUID,
        db: AsyncSession,
        job_id: Optional[uuid.UUID] = None,
        force_regenerate: bool = False,
        progress_callback: Optional[callable] = None
    ) -> EmbeddingResult:
        """
        Generate embeddings for all chunks in a document.
        
        Args:
            document_id: Document ID
            project_id: Project ID
            db: Database session
            job_id: Optional job ID for progress updates
            force_regenerate: Force regeneration even if cached
            progress_callback: Optional callback for progress
            
        Returns:
            EmbeddingResult with statistics
        """
        start_time = time.time()
        
        # Get all chunks for this document
        chunks = await repositories.get_chunks_by_document(db, document_id)
        
        if not chunks:
            return EmbeddingResult(
                document_id=str(document_id),
                project_id=str(project_id),
                total_chunks=0,
                embedded_chunks=0,
                cached_chunks=0,
                total_tokens=0,
                processing_time=0,
                is_complete=True
            )
        
        # Update job progress
        if job_id:
            await self._update_job_progress(db, job_id, 10, "Loading chunks...")
        
        # Prepare input chunks
        input_chunks = []
        for chunk in chunks:
            metadata = chunk.chunk_metadata or {}
            content_hash = metadata.get("content_hash", "")
            
            input_chunks.append(EmbeddingChunkInput(
                chunk_id=str(chunk.id),
                content=chunk.content,
                content_hash=content_hash,
                metadata=metadata
            ))
        
        total_chunks = len(input_chunks)
        total_batches = (total_chunks + self.config.batch_size - 1) // self.config.batch_size
        
        cached_count = 0
        embedded_count = 0
        total_tokens = 0
        
        # Process in batches
        for batch_num in range(total_batches):
            start_idx = batch_num * self.config.batch_size
            end_idx = min(start_idx + self.config.batch_size, total_chunks)
            batch = input_chunks[start_idx:end_idx]
            
            # Update progress
            if job_id:
                await self._update_job_progress(
                    db, job_id,
                    10 + int((batch_num / total_batches) * 80),
                    f"Processing batch {batch_num + 1} of {total_batches}"
                )
            
            try:
                # Check cache
                cached_chunks, uncached_chunks = await self._check_cache(batch, db)
                cached_count += len(cached_chunks)
                
                # Generate embeddings for uncached chunks
                if uncached_chunks and not force_regenerate:
                    # Retry logic for failed batches
                    for retry in range(self.config.max_retries):
                        try:
                            new_embeddings = await self._generate_embeddings_batch(uncached_chunks)
                            
                            # Update database
                            await self._update_chunks_with_embeddings(db, new_embeddings)
                            embedded_count += len(uncached_chunks)
                            
                            break
                        except Exception as e:
                            if retry == self.config.max_retries - 1:
                                # Log error but continue
                                logger.warning("Failed to generate embeddings for batch %s: %s", batch_num, e)
                            else:
                                await asyncio.sleep(self.config.retry_delay_seconds)
                
                # Calculate tokens
                for chunk in batch:
                    if chunk.metadata and "token_count" in chunk.metadata:
                        total_tokens += chunk.metadata["token_count"]
                
                # Call progress callback
                if progress_callback:
                    progress = EmbeddingProgress(
                        document_id=str(document_id),
                        current_batch=batch_num + 1,
                        total_batches=total_batches,
                        chunks_processed=end_idx,
                        total_chunks=total_chunks,
                        status="embedding",
                        progress_percent=((batch_num + 1) / total_batches) * 100,
                        message=f"Processed {end_idx}/{total_chunks} chunks",
                        cached_count=cached_count
                    )
                    await progress_callback(progress)
                    
            except Exception as e:
                logger.warning("Error processing batch %s: %s", batch_num, e)
                # Continue with next batch
        
        # Final progress
        if job_id:
            await self._update_job_progress(
                db, job_id, 100,
                f"Completed: {embedded_count} new, {cached_count} cached"
            )
        
        processing_time = time.time() - start_time
        
        return EmbeddingResult(
            document_id=str(document_id),
            project_id=str(project_id),
            total_chunks=total_chunks,
            embedded_chunks=embedded_count + cached_count,
            cached_chunks=cached_count,
            total_tokens=total_tokens,
            processing_time=processing_time,
            is_complete=True
        )
    
    async def generate_embeddings_incremental(
        self,
        document_id: uuid.UUID,
        project_id: uuid.UUID,
        db: AsyncSession,
        job_id: Optional[uuid.UUID] = None
    ) -> EmbeddingResult:
        """
        Generate embeddings only for chunks that don't have them.
        
        This is more efficient for adding new chunks to existing documents.
        """
        # Get chunks without embeddings
        result = await db.execute(
            select(Chunk).where(
                Chunk.document_id == document_id,
                Chunk.embedding.is_(None)
            )
        )
        chunks_without_embeddings = result.scalars().all()
        
        if not chunks_without_embeddings:
            return EmbeddingResult(
                document_id=str(document_id),
                project_id=str(project_id),
                total_chunks=0,
                embedded_chunks=0,
                cached_chunks=0,
                total_tokens=0,
                processing_time=0,
                is_complete=True
            )
        
        # Prepare input
        input_chunks = []
        for chunk in chunks_without_embeddings:
            metadata = chunk.chunk_metadata or {}
            content_hash = metadata.get("content_hash", "")
            
            input_chunks.append(EmbeddingChunkInput(
                chunk_id=str(chunk.id),
                content=chunk.content,
                content_hash=content_hash,
                metadata=metadata
            ))
        
        # Generate embeddings
        start_time = time.time()
        embedded_count = 0
        total_tokens = 0
        
        total_batches = (len(input_chunks) + self.config.batch_size - 1) // self.config.batch_size
        
        for batch_num in range(total_batches):
            start_idx = batch_num * self.config.batch_size
            end_idx = min(start_idx + self.config.batch_size, len(input_chunks))
            batch = input_chunks[start_idx:end_idx]
            
            try:
                new_embeddings = await self._generate_embeddings_batch(batch)
                await self._update_chunks_with_embeddings(db, new_embeddings)
                embedded_count += len(batch)
                
                # Calculate tokens
                for chunk in batch:
                    if chunk.metadata and "token_count" in chunk.metadata:
                        total_tokens += chunk.metadata["token_count"]
                        
            except Exception as e:
                logger.warning("Error in incremental embedding batch %s: %s", batch_num, e)
        
        processing_time = time.time() - start_time
        
        return EmbeddingResult(
            document_id=str(document_id),
            project_id=str(project_id),
            total_chunks=len(chunks_without_embeddings),
            embedded_chunks=embedded_count,
            cached_chunks=0,
            total_tokens=total_tokens,
            processing_time=processing_time,
            is_complete=True
        )


# Default service instance
embedding_service = LocalEmbeddingService()