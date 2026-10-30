"""
Storage Manager for Document Ingestion

Handles file storage with organized directory structure by user/project.
Provides both standard and streaming upload capabilities.
"""

import os
import uuid
import aiofiles
import aiofiles.os
from pathlib import Path
from typing import Optional, BinaryIO
from datetime import datetime


class StorageManager:
    """
    Manages file storage with structured directory organization.
    
    Directory structure: {base_path}/{user_id}/{project_id}/{filename}
    """
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize storage manager.
        
        Args:
            base_path: Base directory for file storage. 
                      Defaults to STORAGE_PATH environment variable or ./storage
        """
        self.base_path = Path(base_path or os.getenv("STORAGE_PATH", "./storage"))
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def _get_project_path(self, user_id: uuid.UUID, project_id: uuid.UUID) -> Path:
        """Get the storage path for a project"""
        project_path = self.base_path / str(user_id) / str(project_id)
        project_path.mkdir(parents=True, exist_ok=True)
        return project_path
    
    def _generate_unique_filename(self, project_path: Path, original_filename: str) -> str:
        """
        Generate a unique filename to avoid conflicts.
        Format: {timestamp}_{uuid}_{original_name}
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        
        # Get file extension
        ext = Path(original_filename).suffix
        name = Path(original_filename).stem
        
        # Create unique name
        return f"{timestamp}_{unique_id}_{name}{ext}"
    
    async def save_file(
        self,
        user_id: uuid.UUID,
        project_id: uuid.UUID,
        file_data: bytes,
        original_filename: str
    ) -> str:
        """
        Save file data to storage (for standard uploads).
        
        Args:
            user_id: User ID
            project_id: Project ID
            file_data: File content as bytes
            original_filename: Original filename from upload
            
        Returns:
            Relative storage path (stored in database)
        """
        project_path = self._get_project_path(user_id, project_id)
        unique_filename = self._generate_unique_filename(project_path, original_filename)
        file_path = project_path / unique_filename
        
        # Write file
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(file_data)
        
        # Return relative path from base_path
        return str(file_path.relative_to(self.base_path))
    
    async def save_stream(
        self,
        user_id: uuid.UUID,
        project_id: uuid.UUID,
        original_filename: str,
        file_stream: BinaryIO,
        chunk_size: int = 64 * 1024  # 64KB chunks
    ) -> tuple[str, int]:
        """
        Save file from stream (for large file uploads).
        
        Args:
            user_id: User ID
            project_id: Project ID
            original_filename: Original filename from upload
            file_stream: File stream to read from
            chunk_size: Size of chunks to read
            
        Returns:
            Tuple of (relative storage path, total bytes written)
        """
        project_path = self._get_project_path(user_id, project_id)
        unique_filename = self._generate_unique_filename(project_path, original_filename)
        file_path = project_path / unique_filename
        
        total_bytes = 0
        
        # Write file in chunks
        async with aiofiles.open(file_path, "wb") as f:
            while True:
                chunk = file_stream.read(chunk_size)
                if not chunk:
                    break
                await f.write(chunk)
                total_bytes += len(chunk)
        
        # Return relative path from base_path
        return str(file_path.relative_to(self.base_path)), total_bytes
    
    async def delete_file(self, relative_path: str) -> bool:
        """
        Delete a file from storage.
        
        Args:
            relative_path: Relative path from base_path
            
        Returns:
            True if deleted, False if not found
        """
        file_path = self.base_path / relative_path
        
        try:
            if file_path.exists():
                await aiofiles.os.remove(file_path)
                return True
        except Exception:
            pass
        
        return False
    
    def get_absolute_path(self, relative_path: str) -> Path:
        """Get absolute path for a file"""
        return self.base_path / relative_path
    
    async def file_exists(self, relative_path: str) -> bool:
        """Check if file exists"""
        file_path = self.base_path / relative_path
        return file_path.exists()
    
    async def get_file_size(self, relative_path: str) -> int:
        """Get file size in bytes"""
        file_path = self.base_path / relative_path
        if file_path.exists():
            stat = await aiofiles.os.stat(file_path)
            return stat.st_size
        return 0
    
    async def cleanup_project_files(self, user_id: uuid.UUID, project_id: uuid.UUID) -> int:
        """
        Clean up all files for a project.
        
        Returns:
            Number of files deleted
        """
        project_path = self.base_path / str(user_id) / str(project_id)
        count = 0
        
        if project_path.exists():
            for file in project_path.iterdir():
                if file.is_file():
                    try:
                        await aiofiles.os.remove(file)
                        count += 1
                    except Exception:
                        pass
        
        return count


# Global storage manager instance
storage_manager = StorageManager()


__all__ = ["StorageManager", "storage_manager"]