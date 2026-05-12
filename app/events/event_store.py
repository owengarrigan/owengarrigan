import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from threading import Lock


@dataclass(frozen=True)
class StoredEvent:
    """Event record returned by the SQLite event store."""

    id: int
    timestamp: str
    camera_name: str
    event_type: str
    confidence: float
    snapshot_path: str
    labels_detected: list[str]


class EventStore:
    """Small SQLite repository for local edge events."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self._lock = Lock()

    def initialize(self) -> None:
        """Create the events table if it does not exist."""

        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    camera_name TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    snapshot_path TEXT NOT NULL,
                    labels_detected TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def add_event(
        self,
        *,
        timestamp: str,
        camera_name: str,
        event_type: str,
        confidence: float,
        snapshot_path: str,
        labels_detected: list[str],
    ) -> StoredEvent:
        """Persist an event and return the stored row."""

        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO events (
                    timestamp,
                    camera_name,
                    event_type,
                    confidence,
                    snapshot_path,
                    labels_detected
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    camera_name,
                    event_type,
                    confidence,
                    snapshot_path,
                    json.dumps(labels_detected),
                ),
            )
            connection.commit()
            event_id = int(cursor.lastrowid)

        stored = self.get_event(event_id)
        if stored is None:
            raise RuntimeError(f"Stored event {event_id} could not be reloaded")
        return stored

    def get_recent(self, limit: int = 50) -> list[StoredEvent]:
        """Return the newest events first."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, timestamp, camera_name, event_type, confidence,
                       snapshot_path, labels_detected
                FROM events
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def get_event(self, event_id: int) -> StoredEvent | None:
        """Return one event by id."""

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, timestamp, camera_name, event_type, confidence,
                       snapshot_path, labels_detected
                FROM events
                WHERE id = ?
                """,
                (event_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_event(row)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    def _row_to_event(self, row: sqlite3.Row) -> StoredEvent:
        return StoredEvent(
            id=int(row["id"]),
            timestamp=str(row["timestamp"]),
            camera_name=str(row["camera_name"]),
            event_type=str(row["event_type"]),
            confidence=float(row["confidence"]),
            snapshot_path=str(row["snapshot_path"]),
            labels_detected=json.loads(str(row["labels_detected"])),
        )
