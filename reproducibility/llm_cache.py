"""
LLM Response Cache System for FBA-Bench Reproducibility

Provides deterministic LLM response caching to ensure scientific reproducibility
by eliminating non-deterministic behavior from external LLM API calls.
"""

import os
import json
import gzip
import hashlib
import logging
import time
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import sqlite3
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@dataclass
class CachedResponse:
    """Represents a cached LLM response with metadata."""
    prompt_hash: str
    response: Dict[str, Any]
    model: str
    temperature: float
    timestamp: str
    metadata: Dict[str, Any]
    response_hash: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CachedResponse':
        """Create from dictionary after deserialization."""
        return cls(**data)

@dataclass
class CacheStatistics:
    """Cache performance and usage statistics."""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_size: int = 0
    last_access: Optional[str] = None
    
    @property
    def hit_ratio(self) -> float:
        """Calculate cache hit ratio."""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests
    
    @property
    def miss_ratio(self) -> float:
        """Calculate cache miss ratio."""
        return 1.0 - self.hit_ratio

class LLMResponseCache:
    """
    Thread-safe LLM response cache for deterministic simulation reproduction.
    
    Supports both in-memory caching for performance and persistent storage
    for cross-session reproducibility. Uses SQLite for reliable persistence
    with optional compression for large responses.
    """
    
    def __init__(
        self,
        cache_file: str = "llm_responses.cache",
        enable_compression: bool = True,
        enable_validation: bool = True,
        max_memory_entries: int = 10000,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize LLM response cache.
        
        Args:
            cache_file: Path to persistent cache file
            enable_compression: Whether to compress large responses
            enable_validation: Whether to validate cache integrity
            max_memory_entries: Maximum entries to keep in memory
            cache_dir: Directory for cache files (default: reproducibility/cache)
        """
        self.enable_compression = enable_compression
        self.enable_validation = enable_validation
        self.max_memory_entries = max_memory_entries
        
        # Setup cache directory
        if cache_dir is None:
            cache_dir = Path(__file__).parent / "cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_file = self.cache_dir / cache_file
        
        # In-memory cache for performance
        self._memory_cache: Dict[str, CachedResponse] = {}
        self._access_order: List[str] = []  # For LRU eviction
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self._stats = CacheStatistics()
        
        # Operating modes
        self._deterministic_mode = False
        self._recording_mode = False
        
        # Initialize persistent storage
        self._init_database()
        
        logger.info(f"LLM cache initialized: {self.cache_file}")
    
    def _init_database(self):
        """Initialize SQLite database for persistent storage."""
        with sqlite3.connect(self.cache_file, timeout=30.0) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS llm_responses (
                    prompt_hash TEXT PRIMARY KEY,
                    response_data BLOB NOT NULL,
                    model TEXT NOT NULL,
                    temperature REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT,
                    response_hash TEXT NOT NULL,
                    compressed INTEGER DEFAULT 0
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Create indexes for performance
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_temp 
                ON llm_responses(model, temperature)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON llm_responses(timestamp)
            """)
            
            conn.commit()
    
    @contextmanager
    def _db_connection(self):
        """Get database connection with proper error handling."""
        conn = None
        try:
            conn = sqlite3.connect(self.cache_file, timeout=30.0)
            conn.row_factory = sqlite3.Row
            yield conn
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()
    
    def generate_prompt_hash(
        self,
        prompt: str,
        model: str,
        temperature: float,
        **kwargs
    ) -> str:
        """
        Generate deterministic hash for prompt and parameters.
        
        Args:
            prompt: The input prompt
            model: Model name
            temperature: Sampling temperature
            **kwargs: Additional parameters that affect output
            
        Returns:
            Deterministic hash string
        """
        # Create canonical representation
        hash_data = {
            "prompt": prompt,
            "model": model,
            "temperature": round(temperature, 6),  # Normalize float precision
            "params": dict(sorted(kwargs.items()))
        }
        
        # Convert to canonical JSON string
        canonical_str = json.dumps(hash_data, sort_keys=True, separators=(',', ':'))
        
        # Generate SHA256 hash
        return hashlib.sha256(canonical_str.encode('utf-8')).hexdigest()
    
    def _generate_response_hash(self, response: Dict[str, Any]) -> str:
        """Generate hash of response content for integrity checking."""
        response_str = json.dumps(response, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(response_str.encode('utf-8')).hexdigest()
    
    def _compress_data(self, data: bytes) -> bytes:
        """Compress data if compression is enabled."""
        if self.enable_compression and len(data) > 1024:  # Only compress larger responses
            return gzip.compress(data)
        return data
    
    def _decompress_data(self, data: bytes, is_compressed: bool) -> bytes:
        """Decompress data if it was compressed."""
        if is_compressed:
            return gzip.decompress(data)
        return data
    
    def _update_memory_cache(self, prompt_hash: str, cached_response: CachedResponse):
        """Update in-memory cache with LRU eviction."""
        with self._lock:
            # Remove from current position if exists
            if prompt_hash in self._access_order:
                self._access_order.remove(prompt_hash)
            
            # Add to end (most recent)
            self._access_order.append(prompt_hash)
            self._memory_cache[prompt_hash] = cached_response
            
            # Evict LRU entries if over limit
            while len(self._memory_cache) > self.max_memory_entries:
                lru_hash = self._access_order.pop(0)
                del self._memory_cache[lru_hash]
    
    def set_deterministic_mode(self, enabled: bool):
        """
        Enable or disable deterministic mode.
        
        In deterministic mode, only cached responses are returned.
        Cache misses will raise an exception.
        """
        with self._lock:
            self._deterministic_mode = enabled
            logger.info(f"Deterministic mode: {'enabled' if enabled else 'disabled'}")
    
    def set_recording_mode(self, enabled: bool):
        """
        Enable or disable recording mode.
        
        In recording mode, all LLM responses are automatically cached.
        """
        with self._lock:
            self._recording_mode = enabled
            logger.info(f"Recording mode: {'enabled' if enabled else 'disabled'}")
    
    def cache_response(
        self,
        prompt_hash: str,
        response: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store LLM response in cache.
        
        Args:
            prompt_hash: Deterministic hash of the prompt
            response: LLM response data
            metadata: Additional metadata to store
            
        Returns:
            True if successfully cached, False otherwise
        """
        try:
            with self._lock:
                timestamp = datetime.now(timezone.utc).isoformat()
                response_hash = self._generate_response_hash(response)
                
                cached_response = CachedResponse(
                    prompt_hash=prompt_hash,
                    response=response,
                    model=metadata.get('model', 'unknown') if metadata else 'unknown',
                    temperature=metadata.get('temperature', 0.0) if metadata else 0.0,
                    timestamp=timestamp,
                    metadata=metadata or {},
                    response_hash=response_hash
                )
                
                # Store in memory cache
                self._update_memory_cache(prompt_hash, cached_response)
                
                # Store in persistent cache
                response_json = json.dumps(response, separators=(',', ':')).encode('utf-8')
                compressed_data = self._compress_data(response_json)
                is_compressed = len(compressed_data) < len(response_json)
                
                with self._db_connection() as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO llm_responses 
                        (prompt_hash, response_data, model, temperature, timestamp, metadata, response_hash, compressed)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        prompt_hash,
                        compressed_data,
                        cached_response.model,
                        cached_response.temperature,
                        timestamp,
                        json.dumps(metadata or {}),
                        response_hash,
                        1 if is_compressed else 0
                    ))
                    conn.commit()
                
                self._stats.cache_size += 1
                logger.debug(f"Cached response for hash: {prompt_hash[:16]}...")
                return True
                
        except Exception as e:
            logger.error(f"Failed to cache response: {e}")
            return False
    
    def get_cached_response(self, prompt_hash: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached LLM response.
        
        Args:
            prompt_hash: Deterministic hash of the prompt
            
        Returns:
            Cached response data or None if not found
        """
        try:
            with self._lock:
                self._stats.total_requests += 1
                self._stats.last_access = datetime.now(timezone.utc).isoformat()
                
                # Check memory cache first
                if prompt_hash in self._memory_cache:
                    cached_response = self._memory_cache[prompt_hash]
                    
                    # Move to end of access order (LRU)
                    self._access_order.remove(prompt_hash)
                    self._access_order.append(prompt_hash)
                    
                    self._stats.cache_hits += 1
                    logger.debug(f"Memory cache hit for hash: {prompt_hash[:16]}...")
                    return cached_response.response
                
                # Check persistent cache
                with self._db_connection() as conn:
                    cursor = conn.execute("""
                        SELECT response_data, model, temperature, timestamp, metadata, response_hash, compressed
                        FROM llm_responses WHERE prompt_hash = ?
                    """, (prompt_hash,))
                    
                    row = cursor.fetchone()
                    if row:
                        # Decompress if needed
                        response_data = self._decompress_data(row['response_data'], bool(row['compressed']))
                        response = json.loads(response_data.decode('utf-8'))
                        
                        # Validate response integrity if enabled
                        if self.enable_validation:
                            stored_hash = row['response_hash']
                            computed_hash = self._generate_response_hash(response)
                            if stored_hash != computed_hash:
                                logger.error(f"Cache corruption detected for hash: {prompt_hash[:16]}...")
                                self._stats.cache_misses += 1
                                return None
                        
                        # Create cached response object
                        cached_response = CachedResponse(
                            prompt_hash=prompt_hash,
                            response=response,
                            model=row['model'],
                            temperature=row['temperature'],
                            timestamp=row['timestamp'],
                            metadata=json.loads(row['metadata']),
                            response_hash=row['response_hash']
                        )
                        
                        # Update memory cache
                        self._update_memory_cache(prompt_hash, cached_response)
                        
                        self._stats.cache_hits += 1
                        logger.debug(f"Persistent cache hit for hash: {prompt_hash[:16]}...")
                        return response
                
                # Cache miss
                self._stats.cache_misses += 1
                
                # In deterministic mode, cache misses are errors
                if self._deterministic_mode:
                    raise ValueError(f"Cache miss in deterministic mode for hash: {prompt_hash[:16]}...")
                
                logger.debug(f"Cache miss for hash: {prompt_hash[:16]}...")
                return None
                
        except Exception as e:
            logger.error(f"Failed to retrieve cached response: {e}")
            if self._deterministic_mode:
                raise
            return None
    
    def validate_cache_integrity(self) -> Tuple[bool, List[str]]:
        """
        Validate integrity of entire cache.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        try:
            with self._db_connection() as conn:
                cursor = conn.execute("""
                    SELECT prompt_hash, response_data, response_hash, compressed
                    FROM llm_responses
                """)
                
                for row in cursor:
                    try:
                        # Decompress and parse response
                        response_data = self._decompress_data(row['response_data'], bool(row['compressed']))
                        response = json.loads(response_data.decode('utf-8'))
                        
                        # Validate hash
                        stored_hash = row['response_hash']
                        computed_hash = self._generate_response_hash(response)
                        
                        if stored_hash != computed_hash:
                            errors.append(f"Hash mismatch for {row['prompt_hash'][:16]}...")
                            
                    except Exception as e:
                        errors.append(f"Parse error for {row['prompt_hash'][:16]}...: {e}")
            
            is_valid = len(errors) == 0
            if is_valid:
                logger.info("Cache integrity validation passed")
            else:
                logger.error(f"Cache integrity validation failed with {len(errors)} errors")
                
            return is_valid, errors
            
        except Exception as e:
            logger.error(f"Cache validation failed: {e}")
            return False, [f"Validation error: {e}"]
    
    def export_cache(self, filepath: str, compress: bool = True) -> bool:
        """
        Export cache to file for sharing.
        
        Args:
            filepath: Destination file path
            compress: Whether to compress the export
            
        Returns:
            True if successful, False otherwise
        """
        try:
            export_data = {
                "version": "1.0",
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "entries": []
            }
            
            with self._db_connection() as conn:
                cursor = conn.execute("""
                    SELECT prompt_hash, response_data, model, temperature, timestamp, metadata, response_hash, compressed
                    FROM llm_responses
                """)
                
                for row in cursor:
                    # Decompress response data
                    response_data = self._decompress_data(row['response_data'], bool(row['compressed']))
                    response = json.loads(response_data.decode('utf-8'))
                    
                    entry = {
                        "prompt_hash": row['prompt_hash'],
                        "response": response,
                        "model": row['model'],
                        "temperature": row['temperature'],
                        "timestamp": row['timestamp'],
                        "metadata": json.loads(row['metadata']),
                        "response_hash": row['response_hash']
                    }
                    export_data["entries"].append(entry)
            
            # Write to file
            export_json = json.dumps(export_data, separators=(',', ':')).encode('utf-8')
            
            if compress:
                export_json = gzip.compress(export_json)
                filepath = f"{filepath}.gz" if not filepath.endswith('.gz') else filepath
            
            with open(filepath, 'wb') as f:
                f.write(export_json)
            
            logger.info(f"Cache exported to: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export cache: {e}")
            return False
    
    def import_cache(self, filepath: str, merge: bool = True) -> bool:
        """
        Import cache from file.
        
        Args:
            filepath: Source file path
            merge: Whether to merge with existing cache or replace
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Read file
            with open(filepath, 'rb') as f:
                data = f.read()
            
            # Decompress if needed
            if filepath.endswith('.gz'):
                data = gzip.decompress(data)
            
            import_data = json.loads(data.decode('utf-8'))
            
            # Validate import format
            if not all(key in import_data for key in ["version", "entries"]):
                raise ValueError("Invalid cache export format")
            
            imported_count = 0
            
            with self._db_connection() as conn:
                if not merge:
                    # Clear existing cache
                    conn.execute("DELETE FROM llm_responses")
                    self._memory_cache.clear()
                    self._access_order.clear()
                
                for entry in import_data["entries"]:
                    # Validate entry
                    if self.enable_validation:
                        computed_hash = self._generate_response_hash(entry["response"])
                        if computed_hash != entry["response_hash"]:
                            logger.warning(f"Skipping corrupted entry: {entry['prompt_hash'][:16]}...")
                            continue
                    
                    # Compress response data
                    response_json = json.dumps(entry["response"], separators=(',', ':')).encode('utf-8')
                    compressed_data = self._compress_data(response_json)
                    is_compressed = len(compressed_data) < len(response_json)
                    
                    # Insert into database
                    conn.execute("""
                        INSERT OR REPLACE INTO llm_responses 
                        (prompt_hash, response_data, model, temperature, timestamp, metadata, response_hash, compressed)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        entry["prompt_hash"],
                        compressed_data,
                        entry["model"],
                        entry["temperature"],
                        entry["timestamp"],
                        json.dumps(entry["metadata"]),
                        entry["response_hash"],
                        1 if is_compressed else 0
                    ))
                    
                    imported_count += 1
                
                conn.commit()
            
            logger.info(f"Imported {imported_count} cache entries from: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import cache: {e}")
            return False
    
    def get_cache_statistics(self) -> CacheStatistics:
        """Get cache performance statistics."""
        with self._lock:
            # Update cache size from database
            try:
                with self._db_connection() as conn:
                    cursor = conn.execute("SELECT COUNT(*) FROM llm_responses")
                    self._stats.cache_size = cursor.fetchone()[0]
            except Exception as e:
                logger.error(f"Failed to get cache size: {e}")
            
            return CacheStatistics(
                total_requests=self._stats.total_requests,
                cache_hits=self._stats.cache_hits,
                cache_misses=self._stats.cache_misses,
                cache_size=self._stats.cache_size,
                last_access=self._stats.last_access
            )
    
    def clear_cache(self, confirm: bool = False) -> bool:
        """
        Clear all cached responses.
        
        Args:
            confirm: Safety confirmation flag
            
        Returns:
            True if cleared, False otherwise
        """
        if not confirm:
            logger.warning("Cache clear operation requires confirmation flag")
            return False
        
        try:
            with self._lock:
                # Clear memory cache
                self._memory_cache.clear()
                self._access_order.clear()
                
                # Clear persistent cache
                with self._db_connection() as conn:
                    conn.execute("DELETE FROM llm_responses")
                    conn.commit()
                
                # Reset statistics
                self._stats = CacheStatistics()
                
                logger.info("Cache cleared successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Any cleanup if needed
        pass