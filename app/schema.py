"""
Shared data types. A Chunk always remembers where it came from so every
answer can cite a source URL.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List


@dataclass
class Chunk:
    text: str
    url: str = ""
    title: str = ""
    section: str = ""
    id: str = field(default="")

    def __post_init__(self):
        if not self.id:
            digest = hashlib.sha1(f"{self.url}|{self.section}|{self.text}".encode()).hexdigest()
            self.id = digest[:12]

    @property
    def header(self) -> str:
        """Title + section, prepended to the text the retrievers see."""
        parts = [p for p in (self.title, self.section) if p]
        return " > ".join(parts)

    @property
    def indexed_text(self) -> str:
        return f"{self.header}\n{self.text}" if self.header else self.text


@dataclass
class Hit:
    chunk: Chunk
    score: float


def write_chunks(chunks: Iterable[Chunk], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_chunks(path: str | Path) -> List[Chunk]:
    with Path(path).open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]
