"""ALRI Record Serialization and Storage Module.

This module provides classes for serializing, deserializing, and persisting
ALRI (Aadhaar Lifecycle Risk Index) records to JSON format.

Requirements: 13.1, 13.2, 13.3, 13.4
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple


@dataclass
class ALRIRecord:
    """Data class representing an ALRI computation record.
    
    Attributes:
        district: District name
        state: State name
        alri_score: Composite ALRI score (0-100)
        sub_scores: Dictionary of sub-scores (coverage, instability, biometric, anomaly)
        reason_codes: List of reason code strings
        computed_at: ISO format timestamp of computation
    """
    district: str
    state: str
    alri_score: float
    sub_scores: Dict[str, float] = field(default_factory=dict)
    reason_codes: List[str] = field(default_factory=list)
    computed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __eq__(self, other: object) -> bool:
        """Check equality with another ALRIRecord."""
        if not isinstance(other, ALRIRecord):
            return False
        return (
            self.district == other.district and
            self.state == other.state and
            abs(self.alri_score - other.alri_score) < 1e-9 and
            self._sub_scores_equal(other.sub_scores) and
            self.reason_codes == other.reason_codes and
            self.computed_at == other.computed_at
        )
    
    def _sub_scores_equal(self, other_sub_scores: Dict[str, float]) -> bool:
        """Compare sub_scores with floating point tolerance."""
        if set(self.sub_scores.keys()) != set(other_sub_scores.keys()):
            return False
        for key in self.sub_scores:
            if abs(self.sub_scores[key] - other_sub_scores[key]) >= 1e-9:
                return False
        return True


class ALRISerializer:
    """Serializes and deserializes ALRI records to/from JSON.
    
    Provides methods for converting ALRIRecord objects to JSON strings
    and vice versa, ensuring data integrity through round-trip operations.
    
    Requirements: 13.2, 13.4
    """
    
    def serialize(self, record: ALRIRecord) -> str:
        """Convert ALRIRecord to JSON string.
        
        Args:
            record: ALRIRecord instance to serialize
            
        Returns:
            JSON string representation of the record
            
        Raises:
            SerializationError: If serialization fails
        """
        try:
            record_dict = asdict(record)
            return json.dumps(record_dict, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            raise SerializationError(f"Failed to serialize record: {e}")
    
    def deserialize(self, json_str: str) -> ALRIRecord:
        """Convert JSON string to ALRIRecord.
        
        Args:
            json_str: JSON string to deserialize
            
        Returns:
            ALRIRecord instance
            
        Raises:
            SerializationError: If deserialization fails
        """
        try:
            data = json.loads(json_str)
            return ALRIRecord(
                district=data['district'],
                state=data['state'],
                alri_score=float(data['alri_score']),
                sub_scores={k: float(v) for k, v in data.get('sub_scores', {}).items()},
                reason_codes=list(data.get('reason_codes', [])),
                computed_at=data.get('computed_at', datetime.now().isoformat())
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            raise SerializationError(f"Failed to deserialize record: {e}")


class SerializationError(Exception):
    """Exception raised when serialization or deserialization fails."""
    pass


class ALRIStorage:
    """Persists ALRI records to JSON files.
    
    Provides methods for saving, loading, and querying ALRI records
    from persistent JSON storage.
    
    Requirements: 13.1, 13.2, 13.3
    """
    
    def __init__(self, storage_dir: str = "data/alri_storage"):
        """Initialize storage with directory path.
        
        Args:
            storage_dir: Directory path for storing JSON files
        """
        self.storage_dir = storage_dir
        self.serializer = ALRISerializer()
        self._records: List[ALRIRecord] = []
    
    def save(self, records: List[ALRIRecord], filepath: str) -> None:
        """Save records to JSON file.
        
        Args:
            records: List of ALRIRecord instances to save
            filepath: Path to the output JSON file
            
        Raises:
            StorageError: If write operation fails after retries
        """
        # Ensure directory exists
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        # Serialize all records
        serialized_records = []
        for record in records:
            try:
                record_dict = json.loads(self.serializer.serialize(record))
                serialized_records.append(record_dict)
            except SerializationError as e:
                raise StorageError(f"Failed to serialize record: {e}")
        
        # Write with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(serialized_records, f, indent=2, ensure_ascii=False)
                # Update in-memory cache
                self._records = list(records)
                return
            except IOError as e:
                if attempt == max_retries - 1:
                    raise StorageError(f"Storage write failed after {max_retries} retries: {e}")
    
    def load(self, filepath: str) -> List[ALRIRecord]:
        """Load records from JSON file.
        
        Args:
            filepath: Path to the JSON file to load
            
        Returns:
            List of ALRIRecord instances
            
        Raises:
            FileNotFoundError: If file does not exist
            StorageError: If loading fails
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise StorageError(f"Corrupted JSON file: {e}")
        
        records = []
        for i, record_dict in enumerate(data):
            try:
                json_str = json.dumps(record_dict)
                record = self.serializer.deserialize(json_str)
                records.append(record)
            except SerializationError as e:
                # Log and skip corrupted records
                print(f"Skipped corrupted record at line {i}: {e}")
                continue
        
        self._records = records
        return records
    
    def query(
        self,
        district: Optional[str] = None,
        date_range: Optional[Tuple[str, str]] = None,
        min_score: Optional[float] = None
    ) -> List[ALRIRecord]:
        """Query records by filters.
        
        Args:
            district: Filter by district name (case-insensitive)
            date_range: Tuple of (start_date, end_date) in ISO format
            min_score: Minimum ALRI score threshold
            
        Returns:
            List of ALRIRecord instances matching all filters
        """
        results = self._records
        
        # Filter by district
        if district is not None:
            results = [r for r in results if r.district.lower() == district.lower()]
        
        # Filter by date range
        if date_range is not None:
            start_date, end_date = date_range
            results = [
                r for r in results
                if start_date <= r.computed_at <= end_date
            ]
        
        # Filter by minimum score
        if min_score is not None:
            results = [r for r in results if r.alri_score >= min_score]
        
        return results


class StorageError(Exception):
    """Exception raised when storage operations fail."""
    pass
