"""
Cache Key Generation Utilities

Utilities for generating deterministic, collision-resistant cache keys.
"""

import hashlib
import json
from typing import Any, Dict, List, Optional
import uuid

from .types import CacheType, KEY_PREFIXES


def _normalize_value(value: Any) -> str:
    """Normalize a value for hashing"""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    elif isinstance(value, list):
        return json.dumps(value, sort_keys=True)
    elif isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    elif isinstance(value, uuid.UUID):
        return str(value)
    else:
        return str(value)


def _hash_content(content: str) -> str:
    """Generate SHA256 hash of content"""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _make_key(prefix: str, *parts: str) -> str:
    """Make a cache key from parts"""
    return ":".join([prefix] + [str(p) for p in parts])


# ==================== Embedding Cache Keys ====================

def embedding_cache_key(content: str) -> str:
    """
    Generate cache key for embedding based on content hash.
    
    Args:
        content: Text content to embed
        
    Returns:
        Cache key: "emb:<content_hash>"
    """
    content_hash = _hash_content(content)
    return _make_key(KEY_PREFIXES[CacheType.EMBEDDING], content_hash)


def embedding_cache_key_from_hash(content_hash: str) -> str:
    """
    Generate cache key from pre-computed content hash.
    
    Args:
        content_hash: SHA256 hash of content
        
    Returns:
        Cache key
    """
    return _make_key(KEY_PREFIXES[CacheType.EMBEDDING], content_hash[:16])


# ==================== Generated Output Cache Keys ====================

def generated_output_cache_key(
    chunk_hashes: List[str],
    mode: str,
    config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate cache key for generated output.
    
    Key is based on:
    - Sorted chunk hashes (content context)
    - Generation mode (notes, flashcards, quiz, etc.)
    - Configuration parameters
    
    Args:
        chunk_hashes: List of content hashes for retrieved chunks
        mode: Generation mode
        config: Optional configuration parameters
        
    Returns:
        Cache key
    """
    # Sort hashes for deterministic key
    sorted_hashes = sorted(chunk_hashes)
    
    # Combine into key material
    key_material = "|".join(sorted_hashes) + f"|{mode}"
    
    if config:
        # Add config to key material
        config_str = json.dumps(config, sort_keys=True)
        key_material += f"|{config_str}"
    
    # Hash the key material
    key_hash = hashlib.sha256(key_material.encode("utf-8")).hexdigest()[:24]
    
    return _make_key(KEY_PREFIXES[CacheType.GENERATED_OUTPUT], mode, key_hash)


def generated_output_cache_key_for_document(
    document_id: uuid.UUID,
    mode: str,
    config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate cache key for entire document generation.
    
    Args:
        document_id: Document ID
        mode: Generation mode
        config: Optional configuration
        
    Returns:
        Cache key
    """
    key_material = f"{document_id}|{mode}"
    
    if config:
        config_str = json.dumps(config, sort_keys=True)
        key_material += f"|{config_str}"
    
    key_hash = hashlib.sha256(key_material.encode("utf-8")).hexdigest()[:24]
    
    return _make_key(KEY_PREFIXES[CacheType.GENERATED_OUTPUT], mode, str(document_id)[:8], key_hash)


# ==================== Research Result Cache Keys ====================

def research_cache_key(query: str, filters: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate cache key for research/search results.
    
    Args:
        query: Search query
        filters: Optional filters (num_results, etc.)
        
    Returns:
        Cache key
    """
    # Normalize query
    normalized_query = query.lower().strip()
    query_hash = _hash_content(normalized_query)
    
    key_material = query_hash
    
    if filters:
        # Add filtered parameters
        filter_str = json.dumps(filters, sort_keys=True)
        filter_hash = hashlib.sha256(filter_str.encode("utf-8")).hexdigest()[:8]
        key_material += f"|{filter_hash}"
    
    return _make_key(KEY_PREFIXES[CacheType.RESEARCH_RESULT], key_material)


def research_url_cache_key(url: str) -> str:
    """
    Generate cache key for extracted URL content.
    
    Args:
        url: URL that was extracted
        
    Returns:
        Cache key
    """
    url_hash = _hash_content(url)
    return _make_key(KEY_PREFIXES[CacheType.RESEARCH_RESULT], "url", url_hash)


# ==================== Document Metadata Cache Keys ====================

def document_metadata_cache_key(document_id: uuid.UUID) -> str:
    """
    Generate cache key for document metadata.
    
    Args:
        document_id: Document ID
        
    Returns:
        Cache key
    """
    return _make_key(KEY_PREFIXES[CacheType.DOCUMENT_METADATA], str(document_id))


def document_chunks_cache_key(document_id: uuid.UUID) -> str:
    """
    Generate cache key for document chunks list.
    
    Args:
        document_id: Document ID
        
    Returns:
        Cache key
    """
    return _make_key(KEY_PREFIXES[CacheType.DOCUMENT_METADATA], "chunks", str(document_id))


# ==================== User Session Cache Keys ====================

def user_session_cache_key(session_id: uuid.UUID) -> str:
    """
    Generate cache key for user session data.
    
    Args:
        session_id: Session ID
        
    Returns:
        Cache key
    """
    return _make_key(KEY_PREFIXES[CacheType.USER_SESSION], str(session_id))


# ==================== Retrieval Result Cache Keys ====================

def retrieval_cache_key(
    query: str,
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 5
) -> str:
    """
    Generate cache key for retrieval results.
    
    Args:
        query: Retrieval query
        filters: Optional filters
        top_k: Number of results
        
    Returns:
        Cache key
    """
    normalized_query = query.lower().strip()
    query_hash = _hash_content(normalized_query)
    
    key_material = f"{query_hash}|k{top_k}"
    
    if filters:
        filter_str = json.dumps(filters, sort_keys=True)
        filter_hash = hashlib.sha256(filter_str.encode("utf-8")).hexdigest()[:8]
        key_material += f"|{filter_hash}"
    
    return _make_key(KEY_PREFIXES[CacheType.RETRIEVAL_RESULT], key_material)


# ==================== Project-based Invalidation ====================

def project_cache_prefix(project_id: uuid.UUID) -> str:
    """
    Get cache key prefix for a project.
    
    Used for invalidating all cache entries for a project.
    
    Args:
        project_id: Project ID
        
    Returns:
        Prefix for project keys
    """
    return f"gen:{project_id}"


def document_cache_prefix(document_id: uuid.UUID) -> str:
    """
    Get cache key prefix for a document.
    
    Used for invalidating all cache entries for a document.
    
    Args:
        document_id: Document ID
        
    Returns:
        Prefix for document keys
    """
    return f"emb:{document_id}"


# ==================== Utility Functions ====================

def parse_cache_key(key: str) -> Dict[str, str]:
    """
    Parse a cache key into components.
    
    Args:
        key: Cache key
        
    Returns:
        Dictionary with prefix and parts
    """
    parts = key.split(":")
    if len(parts) < 2:
        return {"prefix": "", "parts": parts}
    
    return {
        "prefix": parts[0],
        "parts": parts[1:],
    }


def estimate_cache_size(value: Any) -> int:
    """
    Estimate the size of a cached value in bytes.
    
    Args:
        value: Value to estimate
        
    Returns:
        Approximate size in bytes
    """
    try:
        serialized = json.dumps(value)
        return len(serialized.encode("utf-8"))
    except Exception:
        return len(str(value).encode("utf-8"))