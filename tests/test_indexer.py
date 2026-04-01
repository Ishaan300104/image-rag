import pytest
from pathlib import Path
from PIL import Image
from unittest.mock import patch, MagicMock
import numpy as np


@pytest.fixture
def fake_model():
    model = MagicMock()
    model.encode.return_value = np.random.rand(512).astype("float32")
    return model


def test_index_empty_directory(tmp_path, capsys):
    with patch("imagerag.indexer._load_model") as mock:
        mock.return_value = MagicMock()
        from imagerag.indexer import index_directory
        index_directory(tmp_path, verbose=False)

    captured = capsys.readouterr()
    assert "No images or videos found" in captured.out


def test_index_single_image(tmp_path, fake_model):
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    img_path = tmp_path / "test.jpg"
    img.save(img_path)

    index_dir = tmp_path / ".imagerag"

    with (
        patch("imagerag.indexer._load_model", return_value=fake_model),
        patch("imagerag.indexer.INDEX_DIR", index_dir),
        patch("imagerag.indexer.INDEX_FILE", index_dir / "index.faiss"),
        patch("imagerag.indexer.META_FILE", index_dir / "metadata.json"),
    ):
        from imagerag.indexer import index_directory
        index_directory(tmp_path, verbose=False)

    assert (index_dir / "metadata.json").exists()
    import json
    metadata = json.loads((index_dir / "metadata.json").read_text())
    assert len(metadata) == 1
    assert metadata[0]["type"] == "image"
    assert metadata[0]["path"] == str(img_path)
