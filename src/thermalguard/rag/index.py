"""Build and query the local vector index over manuals + maintenance logs.

Embeddings: sentence-transformers/all-MiniLM-L6-v2 (local, free). Index: normalised numpy matrix (exact cosine
search; fine for a few thousand chunks, and swappable for FAISS/pgvector at scale).
Usage:  python -m thermalguard.rag.index      (rebuild)
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from functools import lru_cache

import numpy as np

from thermalguard import config

KB = config.ROOT / "docs" / "knowledge"
NASA_PDF = config.RAW_DIR / "Damage Propagation Modeling.pdf"
OUT = config.ARTIFACTS_DIR / "rag"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def chunk(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    paras, chunks, cur = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()], [], ""
    for p in paras:
        if len(cur) + len(p) + 2 <= size:
            cur = f"{cur}\n\n{p}" if cur else p
            continue
        if cur:
            chunks.append(cur)
        while len(p) > size:  # very long paragraph: hard split
            chunks.append(p[:size])
            p = p[size - overlap:]
        cur = (chunks[-1][-overlap:] + "\n\n" + p) if chunks else p
    if cur:
        chunks.append(cur)
    return chunks


def load_documents() -> list[dict]:
    docs = []
    manual = (KB / "turbofan_maintenance_manual.md").read_text(encoding="utf-8")
    for section in re.split(r"\n(?=## )", manual):
        title = section.splitlines()[0].lstrip("# ").strip()
        for c in chunk(section):
            docs.append({"text": c, "source": f"Maintenance manual: {title}", "asset_id": None})

    if NASA_PDF.exists():
        from pypdf import PdfReader

        for i, page in enumerate(PdfReader(str(NASA_PDF)).pages, start=1):
            for c in chunk(page.extract_text() or ""):
                docs.append({"text": c, "source": f"NASA Damage Propagation Modeling (p.{i})", "asset_id": None})

    logs = defaultdict(list)
    for line in (KB / "maintenance_logs.jsonl").read_text().splitlines():
        e = json.loads(line)
        logs[e["asset_id"]].append(f"{e['date']} | cycle {e['cycle']} | {e['technician']} | {e['note']}")
    for asset, entries in logs.items():
        text = f"Maintenance log for engine {asset}:\n" + "\n".join(entries)
        docs.append({"text": text, "source": f"Maintenance log: {asset}", "asset_id": asset})
    return docs


@lru_cache(maxsize=1)
def embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL, device="cpu")


def build() -> None:
    docs = load_documents()
    vecs = embedder().encode([d["text"] for d in docs], normalize_embeddings=True, batch_size=64)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "vectors.npy", vecs.astype(np.float32))
    (OUT / "chunks.json").write_text(json.dumps(docs, indent=1))
    by_src = defaultdict(int)
    for d in docs:
        by_src[d["source"].split(":")[0].split(" (")[0]] += 1
    print(f"indexed {len(docs)} chunks: {dict(by_src)}")


class Retriever:
    def __init__(self):
        self.vecs = np.load(OUT / "vectors.npy")
        self.docs = json.loads((OUT / "chunks.json").read_text())

    def search(self, query: str, k: int = 5, asset_id: str | None = None) -> list[dict]:
        q = embedder().encode([query], normalize_embeddings=True)[0]
        scores = self.vecs @ q
        order = list(np.argsort(-scores)[:k])
        if asset_id:  # always include that engine's own maintenance log
            own = [i for i, d in enumerate(self.docs) if d["asset_id"] == asset_id]
            order = own + [i for i in order if i not in own][: k - len(own)]
        return [{**self.docs[i], "score": float(scores[i])} for i in order]


if __name__ == "__main__":
    build()
