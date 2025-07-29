import os
import hashlib
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import datetime

# Ensure the parent directory for artifacts exists
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

class EventSnapshot:
    """
    Manages the dumping and comparison of simulation event streams.
    Events are stored in Parquet format for efficient storage and retrieval.
    """

    @staticmethod
    def _create_event_df(events: List[Dict[str, Any]]) -> pd.DataFrame:
        """Converts a list of event dictionaries into a Pandas DataFrame."""
        if not events:
            return pd.DataFrame() # Return empty DataFrame if no events
        
        # Ensure all events have a 'timestamp' for sorting and consistency
        for event in events:
            if 'timestamp' not in event:
                event['timestamp'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        df = pd.DataFrame(events)
        # Ensure consistent column order for reproducibility, if possible.
        # This might be tricky with varying event types, but common columns first.
        common_cols = ['timestamp', 'event_type', 'data'] # Assuming these are common
        other_cols = [col for col in df.columns if col not in common_cols]
        df = df[common_cols + sorted(other_cols)]
        return df

    @staticmethod
    def dump_events(events: List[Dict[str, Any]], git_sha: str, run_id: str) -> Path:
        """
        Dumps the given event stream to a parquet file in the artifacts directory.
        Filename format: artifacts/<git_sha>_<run_id>.parquet
        """
        if not events:
            print("Warning: No events to dump. Skipping snapshot creation.")
            return None

        file_name = f"{git_sha}_{run_id}.parquet"
        file_path = ARTIFACTS_DIR / file_name
        
        df = EventSnapshot._create_event_df(events)
        df.to_parquet(file_path, index=False)
        print(f"Event snapshot dumped to: {file_path}")
        return file_path

    @staticmethod
    def load_events(file_path: Path) -> List[Dict[str, Any]]:
        """
        Loads an event stream from a parquet file.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Event snapshot file not found: {file_path}")
        
        df = pd.read_parquet(file_path)
        return df.to_dict(orient='records')

    @staticmethod
    def compare_event_streams(events1: List[Dict[str, Any]], events2: List[Dict[str, Any]]) -> bool:
        """
        Compares two event streams for exact equality.
        This is crucial for golden snapshot testing.
        """
        df1 = EventSnapshot._create_event_df(events1)
        df2 = EventSnapshot._create_event_df(events2)

        # Basic check: same number of rows and columns
        if df1.shape != df2.shape:
            print(f"Shape mismatch: {df1.shape} vs {df2.shape}")
            return False
        
        # Advanced check: compare content ignoring potential minor float differences
        # For exact reproducibility, floating point numbers should be identical too.
        # Using .equals() for robust DataFrame comparison.
        are_equal = df1.equals(df2)
        
        if not are_equal:
            print("Event stream mismatch detected.")
            # For debugging, you might want to show differences:
            # diff = df1.compare(df2) # Requires pandas >= 1.1
            # print("Differences:\n", diff)
        
        return are_equal
    
    @staticmethod
    def generate_event_stream_hash(events: List[Dict[str, Any]]) -> str:
        """
        Generates a deterministic hash of an event stream.
        This is useful for quick integrity checks without loading full data.
        """
        if not events:
            return "empty_event_stream"
        
        df = EventSnapshot._create_event_df(events)
        # Convert DataFrame to a canonical string representation before hashing
        # This requires careful handling for consistent serialization (e.g., sort columns, sort rows)
        canonical_string = df.sort_values(by=df.columns.tolist()).to_json(orient='records', sort_keys=True)
        return hashlib.sha256(canonical_string.encode()).hexdigest()[:16]