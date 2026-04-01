import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path.home() / ".imagerag"
INDEX_FILE = INDEX_DIR / "index.faiss"
META_FILE = INDEX_DIR / "metadata.json"

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("clip-ViT-B-32")
    return _model


def search(query: str, top_k: int = 10) -> list[dict]:
    """
    Search the index with a text query.

    Returns a list of dicts, each with:
      - path (str): absolute file path
      - type (str): "image" or "video"
      - score (float): cosine similarity score (higher = better)
      - timestamp (float): only present for video results, in seconds
    """
    if not INDEX_FILE.exists() or not META_FILE.exists():
        raise FileNotFoundError(
            "No index found. Run `imagerag index <directory>` first."
        )

    model = _get_model()
    emb = model.encode(query, convert_to_numpy=True)
    emb = emb / np.linalg.norm(emb)
    emb = emb.astype("float32").reshape(1, -1)

    index = faiss.read_index(str(INDEX_FILE))
    with open(META_FILE) as f:
        metadata = json.load(f)

    scores, indices = index.search(emb, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        entry = metadata[idx].copy()
        entry["score"] = float(score)
        results.append(entry)

    return results
