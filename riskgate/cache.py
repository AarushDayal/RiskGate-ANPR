"""Disk cache for Gemini crop responses, including failures."""

import hashlib
import json
from pathlib import Path


def key(crop_bytes: bytes, model: str, prompt_version: str) -> str:
    return hashlib.sha256(crop_bytes + b"\0" + model.encode() + b"\0" + prompt_version.encode()).hexdigest()


class Cache:
    def __init__(self, directory: Path):
        self.directory = directory

    def get(self, digest: str) -> dict | None:
        path = self.directory / f"{digest}.json"
        return json.loads(path.read_text()) if path.exists() else None

    def put(self, digest: str, value: dict) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{digest}.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(value) + "\n")
        temp.replace(path)
