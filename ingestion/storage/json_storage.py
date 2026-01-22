"""
Storage Backend for Ingestion Layer

JSON-first storage with optional database backends.
Enables auditing, reprocessing, and temporal analysis.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import uuid4

from ..schemas import PerceptionEvent, PerceptionBatch, AnomalyRecord


class JSONStorageBackend:
    """
    File-based JSON storage for perception events.
    
    Organized by date for efficient retrieval.
    """
    
    def __init__(self, base_path: str = "D:/vision-to-action/data/ingestion"):
        """
        Initialize JSON storage backend.
        
        Args:
            base_path: Base directory for storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories
        self.events_dir = self.base_path / "events"
        self.batches_dir = self.base_path / "batches"
        self.anomalies_dir = self.base_path / "anomalies"
        
        for dir_path in [self.events_dir, self.batches_dir, self.anomalies_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _get_date_dir(self, base_dir: Path, timestamp: datetime) -> Path:
        """Get date-organized directory"""
        date_dir = base_dir / timestamp.strftime("%Y/%m/%d")
        date_dir.mkdir(parents=True, exist_ok=True)
        return date_dir
    
    def store_event(self, event: PerceptionEvent) -> str:
        """
        Store a perception event.
        
        Args:
            event: PerceptionEvent to store
            
        Returns:
            Path to stored file
        """
        date_dir = self._get_date_dir(self.events_dir, event.timestamp)
        filename = f"{event.event_id}.json"
        filepath = date_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(event.model_dump(), f, indent=2, default=str)
        
        return str(filepath)
    
    def load_event(self, event_id: str, date: Optional[datetime] = None) -> Optional[PerceptionEvent]:
        """
        Load a perception event by ID.
        
        Args:
            event_id: Event ID to load
            date: Optional date hint for faster lookup
            
        Returns:
            PerceptionEvent or None if not found
        """
        if date:
            date_dir = self._get_date_dir(self.events_dir, date)
            filepath = date_dir / f"{event_id}.json"
            if filepath.exists():
                with open(filepath, 'r') as f:
                    data = json.load(f)
                return PerceptionEvent(**data)
        
        # Search all dates
        for filepath in self.events_dir.rglob(f"{event_id}.json"):
            with open(filepath, 'r') as f:
                data = json.load(f)
            return PerceptionEvent(**data)
        
        return None
    
    def store_batch(self, batch: PerceptionBatch) -> str:
        """Store a batch of events"""
        date_dir = self._get_date_dir(self.batches_dir, batch.timestamp)
        filename = f"{batch.batch_id}.json"
        filepath = date_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(batch.model_dump(), f, indent=2, default=str)
        
        # Also store individual events
        for event in batch.events:
            self.store_event(event)
        
        return str(filepath)
    
    def store_anomaly(self, anomaly: AnomalyRecord) -> str:
        """Store an anomaly record"""
        date_dir = self._get_date_dir(self.anomalies_dir, anomaly.timestamp)
        filename = f"{anomaly.anomaly_id}.json"
        filepath = date_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(anomaly.model_dump(), f, indent=2, default=str)
        
        return str(filepath)
    
    def query_events(
        self,
        start_date: datetime,
        end_date: datetime,
        task_type: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[PerceptionEvent]:
        """
        Query events by date range.
        
        Args:
            start_date: Start of date range
            end_date: End of date range
            task_type: Optional filter by task type
            limit: Maximum number of results
            
        Returns:
            List of PerceptionEvents
        """
        events = []
        
        # Iterate through date range
        current_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        while current_date <= end_date:
            date_dir = self._get_date_dir(self.events_dir, current_date)
            
            if date_dir.exists():
                for filepath in date_dir.glob("*.json"):
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    event = PerceptionEvent(**data)
                    
                    # Filter by task type if specified
                    if task_type and event.task_type != task_type:
                        continue
                    
                    events.append(event)
                    
                    if limit and len(events) >= limit:
                        return events
            
            # Move to next day
            current_date = current_date.replace(day=current_date.day + 1)
        
        return events
    
    def get_statistics(self, date: datetime) -> Dict[str, Any]:
        """
        Get statistics for a specific date.
        
        Args:
            date: Date to analyze
            
        Returns:
            Statistics dictionary
        """
        date_dir = self._get_date_dir(self.events_dir, date)
        
        if not date_dir.exists():
            return {"date": date.strftime("%Y-%m-%d"), "total_events": 0}
        
        events = []
        for filepath in date_dir.glob("*.json"):
            with open(filepath, 'r') as f:
                data = json.load(f)
            events.append(PerceptionEvent(**data))
        
        # Calculate statistics
        total_detections = sum(len(e.detections) for e in events)
        total_segmentations = sum(len(e.segmentations) for e in events)
        
        task_counts = {}
        for event in events:
            task_counts[event.task_type] = task_counts.get(event.task_type, 0) + 1
        
        return {
            "date": date.strftime("%Y-%m-%d"),
            "total_events": len(events),
            "total_detections": total_detections,
            "total_segmentations": total_segmentations,
            "by_task": task_counts
        }


def generate_event_id() -> str:
    """Generate unique event ID"""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    unique = str(uuid4())[:8]
    return f"evt_{timestamp}_{unique}"


def generate_batch_id() -> str:
    """Generate unique batch ID"""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    unique = str(uuid4())[:8]
    return f"batch_{timestamp}_{unique}"


def generate_anomaly_id() -> str:
    """Generate unique anomaly ID"""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    unique = str(uuid4())[:8]
    return f"anomaly_{timestamp}_{unique}"
