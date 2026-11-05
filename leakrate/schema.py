"""Data model shared by generator, pseudonymizers and metrics (METHODS.md §2–§4)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Iterator

DIRECT_CATEGORIES = {"PER", "CF", "IBAN", "PIVA", "EMAIL", "PHONE", "ADDRESS", "DOC_ID", "PLATE", "DATE_BIRTH"}
QUASI_CATEGORIES = {"ORG", "LOC", "AGE", "PROFESSION", "HEALTH", "DATE"}
ALL_CATEGORIES = DIRECT_CATEGORIES | QUASI_CATEGORIES
DOMAINS = ("legal", "financial", "hr", "clinical")


@dataclass
class Entity:
    id: str                 # coreference id, stable across mentions in a document
    category: str           # one of ALL_CATEGORIES
    kind: str               # "direct" | "quasi"
    start: int
    end: int
    surface: str
    canonical: str
    attrs: dict = field(default_factory=dict)

    def __post_init__(self):
        assert self.category in ALL_CATEGORIES, self.category
        assert self.kind in ("direct", "quasi"), self.kind


@dataclass
class Document:
    doc_id: str
    domain: str
    template_id: str
    split: str
    text: str
    entities: list[Entity]
    quasi_profile: dict
    tasks: dict = field(default_factory=dict)   # gold for downstream tasks (non-PII fields)

    def to_json(self) -> dict:
        d = asdict(self)
        return d

    @staticmethod
    def from_json(d: dict) -> "Document":
        ents = [Entity(**e) for e in d["entities"]]
        return Document(doc_id=d["doc_id"], domain=d["domain"], template_id=d["template_id"],
                        split=d.get("split", "test"), text=d["text"], entities=ents,
                        quasi_profile=d.get("quasi_profile", {}), tasks=d.get("tasks", {}))

    def mentions(self, category: str | None = None, kind: str | None = None) -> list[Entity]:
        out = self.entities
        if category:
            out = [e for e in out if e.category == category]
        if kind:
            out = [e for e in out if e.kind == kind]
        return out


@dataclass
class MappingEntry:
    pseudonym: str
    category: str
    canonical: str
    surfaces: list[str]
    attrs: dict = field(default_factory=dict)


@dataclass
class Mapping:
    doc_id: str
    mode: str                 # "typed" | "opaque"
    entries: list[MappingEntry]

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(d: dict) -> "Mapping":
        return Mapping(doc_id=d["doc_id"], mode=d["mode"], entries=[MappingEntry(**e) for e in d["entries"]])


@dataclass
class Span:
    """Detector output before pseudonym assignment."""
    start: int
    end: int
    category: str
    score: float = 1.0
    text: str = ""


@dataclass
class PseudonymizedDocument:
    doc_id: str
    system: str               # pseudonymizer id
    mode: str                 # typed | opaque
    text: str                 # d'
    mapping: Mapping
    replaced_spans: list[Span]  # spans in the ORIGINAL text that were replaced
    seconds: float = 0.0

    def to_json(self) -> dict:
        return {"doc_id": self.doc_id, "system": self.system, "mode": self.mode, "text": self.text,
                "mapping": self.mapping.to_json(),
                "replaced_spans": [asdict(s) for s in self.replaced_spans], "seconds": self.seconds}

    @staticmethod
    def from_json(d: dict) -> "PseudonymizedDocument":
        return PseudonymizedDocument(doc_id=d["doc_id"], system=d["system"], mode=d["mode"], text=d["text"],
                                     mapping=Mapping.from_json(d["mapping"]),
                                     replaced_spans=[Span(**s) for s in d["replaced_spans"]],
                                     seconds=d.get("seconds", 0.0))


# ---------------------------------------------------------------- JSONL I/O

def write_jsonl(path: str | Path, items: Iterable) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it.to_json() if hasattr(it, "to_json") else it, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_documents(path: str | Path) -> list[Document]:
    return [Document.from_json(d) for d in read_jsonl(path)]


def load_pseudonymized(path: str | Path) -> list[PseudonymizedDocument]:
    return [PseudonymizedDocument.from_json(d) for d in read_jsonl(path)]
