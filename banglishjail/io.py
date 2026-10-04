"""Small JSONL helpers shared by the pipeline scripts."""

import json
import threading
from pathlib import Path


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def latest_by_key(records):
    """Keep one record per `key`: the last successful one, else the last one.

    Resumed runs append retries after failed attempts, so a key can appear
    more than once in a log.
    """
    best = {}
    for rec in records:
        prev = best.get(rec["key"])
        if prev is None or not rec.get("error") or prev.get("error"):
            best[rec["key"]] = rec
    return list(best.values())


class JsonlWriter:
    """Thread-safe append-only JSONL writer that flushes every record."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, record):
        line = json.dumps(record, ensure_ascii=False)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
