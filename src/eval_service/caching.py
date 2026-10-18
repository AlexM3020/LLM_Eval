"""Exact-match response cache.

Key = sha256(model | prompt | generation config). Re-running an eval after a
crash, or comparing two prompt versions against identical model calls, becomes
free and instant. Stretch goal: semantic cache via embedding similarity.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import asdict

from .models.base import GenerationConfig, GenerationResult


class ResponseCache:
    def __init__(self, path: str):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, result_json TEXT NOT NULL)"
        )

    @staticmethod
    def make_key(model: str, prompt: str, config: GenerationConfig) -> str:
        payload = f"{model}|{prompt}|{json.dumps(asdict(config), sort_keys=True, default=str)}"
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, key: str) -> GenerationResult | None:
        row = self.conn.execute("SELECT result_json FROM cache WHERE key = ?", (key,)).fetchone()
        if not row:
            return None
        result = GenerationResult(**json.loads(row[0]))
        result.cached = True
        return result

    def put(self, key: str, result: GenerationResult) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO cache VALUES (?,?)", (key, json.dumps(asdict(result)))
        )
        self.conn.commit()
