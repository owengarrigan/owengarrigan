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

    def search_events(
        self,
        *,
        query: str | None = None,
        camera: str | None = None,
        event_type: str | None = None,
        min_confidence: float | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[StoredEvent]:
        """Search events with flexible filtering."""

        conditions = []
        params: list = []

        if query:
            conditions.append("(labels_detected LIKE ? OR camera_name LIKE ? OR event_type LIKE ?)")
            q = f"%{query}%"
            params.extend([q, q, q])

        if camera:
            conditions.append("camera_name = ?")
            params.append(camera)

        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)

        if min_confidence is not None:
            conditions.append("confidence >= ?")
            params.append(min_confidence)

        where = " AND ".join(conditions) if conditions else "1=1"
        params.extend([limit, offset])

        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT id, timestamp, camera_name, event_type, confidence,
                       snapshot_path, labels_detected
                FROM events
                WHERE {where}
                ORDER BY id DESC
                LIMIT ? OFFSET ?
                """,
                params,
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def get_event_count(self) -> int:
        """Return total number of events."""
        with self._connect() as connection:
            return connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]

    def cleanup_old_events(self, keep_count: int = 1000) -> int:
        """Delete oldest events beyond the retention limit. Return count deleted."""

        with self._lock, self._connect() as connection:
            total = connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            if total <= keep_count:
                return 0

            to_delete = total - keep_count
            rows = connection.execute(
                "SELECT id, snapshot_path FROM events ORDER BY id ASC LIMIT ?",
                (to_delete,),
            ).fetchall()

            deleted_ids = [row["id"] for row in rows]
            snapshot_paths = [row["snapshot_path"] for row in rows]

            connection.execute(
                f"DELETE FROM events WHERE id IN ({','.join('?' * len(deleted_ids))})",
                deleted_ids,
            )
            connection.commit()

            for path_str in snapshot_paths:
                p = Path(path_str)
                if p.exists():
                    p.unlink(missing_ok=True)

            return len(deleted_ids)

    def get_cameras(self) -> list[str]:
        """Return distinct camera names."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT DISTINCT camera_name FROM events ORDER BY camera_name"
            ).fetchall()
        return [row["camera_name"] for row in rows]

    def get_event_types(self) -> list[str]:
        """Return distinct event types."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT DISTINCT event_type FROM events ORDER BY event_type"
            ).fetchall()
        return [row["event_type"] for row in rows]

    def get_stats(self) -> dict:
        """Return aggregate statistics about stored events."""

        with self._connect() as connection:
            total = connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            avg_conf = connection.execute(
                "SELECT AVG(confidence) FROM events"
            ).fetchone()[0]
            rows = connection.execute(
                "SELECT labels_detected FROM events"
            ).fetchall()

        label_counts: dict[str, int] = {}
        for row in rows:
            labels = json.loads(str(row["labels_detected"]))
            for label in labels:
                label_counts[label] = label_counts.get(label, 0) + 1

        return {
            "total_events": total,
            "average_confidence": round(avg_conf or 0, 3),
            "label_counts": label_counts,
        }

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
