"""
JSON-backed store for face embeddings. Keeps everything in memory for fast lookups;
persists to disk on every write so restarts don't lose registered faces.
"""
import json
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingStore:
    def __init__(self, filepath: Path) -> None:
        self._filepath = Path(filepath)
        self._lock = threading.RLock()
        # { name: {name, relationship, note, embeddings: [np.ndarray, ...], avg_embedding: np.ndarray} }
        self._store: Dict[str, dict] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._filepath.exists():
            logger.info("No existing embeddings file — starting fresh.")
            return
        try:
            with open(self._filepath) as f:
                raw = json.load(f)
            for name, data in raw.items():
                data["embeddings"] = [np.array(e, dtype=np.float32) for e in data["embeddings"]]
                data["avg_embedding"] = self._compute_avg(data["embeddings"])
                self._store[name] = data
            logger.info("Loaded %d registered people from %s", len(self._store), self._filepath)
        except Exception:
            logger.warning("embeddings.json is corrupted — deleting and starting fresh.")
            self._filepath.unlink(missing_ok=True)

    def _save(self) -> None:
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        serialisable = {}
        _SKIP = {"embeddings", "avg_embedding"}   # numpy arrays — not JSON-serialisable
        for name, data in self._store.items():
            serialisable[name] = {
                **{k: v for k, v in data.items() if k not in _SKIP},
                "embeddings": [e.tolist() for e in data["embeddings"]],
            }
        with open(self._filepath, "w") as f:
            json.dump(serialisable, f, indent=2)

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def add_embeddings(
        self,
        name: str,
        relationship: str,
        embeddings: List[np.ndarray],
        note: str = "",
    ) -> None:
        """Add (or append to) a person's embedding list."""
        with self._lock:
            if name not in self._store:
                self._store[name] = {
                    "name": name,
                    "relationship": relationship,
                    "note": note,
                    "embeddings": [],
                }
            else:
                # Update metadata in case it changed
                self._store[name]["relationship"] = relationship
                self._store[name]["note"] = note

            self._store[name]["embeddings"].extend(embeddings)
            self._store[name]["avg_embedding"] = self._compute_avg(self._store[name]["embeddings"])
            self._save()
            logger.info(
                "Registered %d embedding(s) for '%s' (total: %d)",
                len(embeddings),
                name,
                len(self._store[name]["embeddings"]),
            )

    def delete_person(self, name: str) -> bool:
        with self._lock:
            if name not in self._store:
                return False
            del self._store[name]
            self._save()
            logger.info("Deleted '%s' from store.", name)
            return True

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_all(self) -> Dict[str, dict]:
        with self._lock:
            return dict(self._store)

    def list_people(self) -> List[dict]:
        with self._lock:
            return [
                {
                    "name": v["name"],
                    "relationship": v["relationship"],
                    "note": v.get("note", ""),
                    "num_embeddings": len(v["embeddings"]),
                }
                for v in self._store.values()
            ]

    def __len__(self) -> int:
        return len(self._store)

    @staticmethod
    def _compute_avg(embeddings: List[np.ndarray]) -> np.ndarray:
        """Average + re-normalize embeddings for fast single-comparison lookup."""
        if not embeddings:
            return np.zeros(512, dtype=np.float32)
        avg = np.mean(embeddings, axis=0).astype(np.float32)
        return avg / (np.linalg.norm(avg) + 1e-6)
