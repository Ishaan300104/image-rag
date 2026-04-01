import pytest
import json
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock


def _write_fake_index(index_dir, n=5):
    import faiss

    dim = 512
    index_dir.mkdir(parents=True, exist_ok=True)
    matrix = np.random.rand(n, dim).astype("float32")
    # normalize
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix /= norms

    idx = faiss.IndexFlatIP(dim)
    idx.add(matrix)
    faiss.write_index(idx, str(index_dir / "index.faiss"))

    metadata = [{"path": f"/fake/image_{i}.jpg", "type": "image"} for i in range(n)]
    (index_dir / "metadata.json").write_text(json.dumps(metadata))


def test_search_returns_results(tmp_path):
    index_dir = tmp_path / ".imagerag"
    _write_fake_index(index_dir)

    fake_model = MagicMock()
    fake_model.encode.return_value = np.random.rand(512).astype("float32")

    with (
        patch("imagerag.searchtool._get_model", return_value=fake_model),
        patch("imagerag.searchtool.INDEX_FILE", index_dir / "index.faiss"),
        patch("imagerag.searchtool.META_FILE", index_dir / "metadata.json"),
    ):
        from imagerag.searchtool import search
        results = search("a test query", top_k=3)

    assert len(results) == 3
    for r in results:
        assert "path" in r
        assert "score" in r
        assert "type" in r


def test_search_raises_without_index(tmp_path):
    with (
        patch("imagerag.searchtool.INDEX_FILE", tmp_path / "index.faiss"),
        patch("imagerag.searchtool.META_FILE", tmp_path / "metadata.json"),
    ):
        from imagerag.searchtool import search
        with pytest.raises(FileNotFoundError):
            search("anything")
