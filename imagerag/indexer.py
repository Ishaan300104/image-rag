import json
import numpy as np
import faiss
import cv2
from pathlib import Path
from PIL import Image
from sentence_transformers import SentenceTransformer
from rich.progress import Progress, BarColumn, TaskProgressColumn, TextColumn, TimeRemainingColumn


# specifying paths
INDEX_DIR = Path.home() / ".imagerag"
INDEX_FILE = INDEX_DIR / "index.faiss"
META_FILE = INDEX_DIR / "metadata.json"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# loading sentence transformer CLIP model
def _load_model():
    return SentenceTransformer("clip-ViT-B-32")

# extracting photos (or frames from videos)
def _extract_video_frames(video_path, interval=5):
    """Yield (PIL Image, timestamp_in_seconds) every `interval` seconds."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return
    frame_interval = int(fps * interval)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / fps
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield Image.fromarray(rgb), timestamp
        frame_idx += 1
    cap.release()


def index_directory(directory, frame_interval=5, verbose=True):
    """
    Scan `directory` recursively, embed all images and video keyframes,
    and save the FAISS index + metadata to ~/.imagerag/.
    """
    INDEX_DIR.mkdir(exist_ok=True)
    model = _load_model()

    embeddings = []
    metadata = []
    directory = Path(directory).expanduser().resolve()

    all_files = [
        p for p in sorted(directory.rglob("*"))
        if p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS
    ]

    if not all_files:
        print("No images or videos found.")
        return

    skipped = 0

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
        TextColumn("[dim]{task.fields[filename]}"),
    ) as progress:
        task = progress.add_task("Indexing", total=len(all_files), filename="")

        for path in all_files:
            suffix = path.suffix.lower()
            progress.update(task, filename=path.name)

            if suffix in IMAGE_EXTS:
                try:
                    img = Image.open(path).convert("RGB")
                    emb = model.encode(img, convert_to_numpy=True)
                    emb = emb / np.linalg.norm(emb)
                    embeddings.append(emb)
                    metadata.append({"path": str(path), "type": "image"})
                except Exception:
                    skipped += 1

            elif suffix in VIDEO_EXTS:
                for frame, ts in _extract_video_frames(path, frame_interval):
                    try:
                        emb = model.encode(frame, convert_to_numpy=True)
                        emb = emb / np.linalg.norm(emb)
                        embeddings.append(emb)
                        metadata.append({"path": str(path), "type": "video", "timestamp": ts})
                    except Exception:
                        skipped += 1

            progress.advance(task)

    if not embeddings:
        print("No images or videos could be indexed.")
        return

    matrix = np.stack(embeddings).astype("float32")
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    faiss.write_index(index, str(INDEX_FILE))
    with open(META_FILE, "w") as f:
        json.dump(metadata, f)

    print(f"\nDone. Indexed {len(embeddings)} items from {directory}")
    if skipped:
        print(f"Skipped {skipped} files due to errors.")
