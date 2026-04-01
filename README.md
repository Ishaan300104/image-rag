# ImageRAG

Search your local photos and videos using text descriptions. Runs fully offline — no internet, no GPU, no cloud.

Built with [CLIP](https://github.com/openai/CLIP) + [FAISS](https://github.com/facebookresearch/faiss) + [Gradio](https://www.gradio.app/).

---

## Features

- Text-to-image search using CLIP (`clip-ViT-B-32`)
- Supports images: JPG, JPEG, PNG, BMP, WEBP, GIF
- Supports videos: MP4, AVI, MOV, MKV, WEBM (searches by keyframe)
- Fully offline — no internet or GPU required
- Works on low-end hardware (tested on Intel i3, 8GB RAM, no GPU)
- Simple web UI and CLI

---

## Requirements

- Python 3.9+
- Linux / macOS / Windows

---

## Installation

**Option 1 — install as a package (recommended):**
```bash
git clone https://github.com/Ishaan300104/image-rag.git
cd image-rag
pip install -e .
```

**Option 2 — install dependencies directly:**
```bash
pip install -r requirements.txt
```

---

## Quick Start

```bash
# 1. Index your photos (only needed once)
imagerag index ~/Pictures

# 2. Search
imagerag search "sunset at beach"

# 3. Or open the web UI
imagerag serve
```

---

## CLI Reference

### `imagerag index <directory>`

Scans a directory recursively and builds a search index. The index is saved to `~/.imagerag/` on your device.

```bash
imagerag index ~/Pictures
```

| Option | Default | Description |
|---|---|---|
| `--frame-interval` | `5` | Seconds between video keyframes extracted for indexing |
| `--quiet` / `-q` | off | Suppress per-file output, only show summary |

**Examples:**
```bash
# Index your Pictures folder
imagerag index ~/Pictures

# Index a custom folder with verbose output (default)
imagerag index /media/external/photos

# Index videos, extracting a frame every 10 seconds
imagerag index ~/Videos --frame-interval 10

# Run quietly, only show final summary
imagerag index ~/Pictures --quiet

# Index multiple folders by running the command again
# (each run rebuilds the full index for that directory)
imagerag index ~/Pictures
imagerag index ~/Downloads/photos
```

> **Note:** Indexing is slow on CPU (~1–2 images/sec). For 1000 photos expect ~10–15 minutes. Run it once and leave it — searching is instant after that.

---

### `imagerag search <query>`

Search the index using a plain text description.

```bash
imagerag search "birthday party"
```

| Option | Default | Description |
|---|---|---|
| `--top` / `-n` | `10` | Number of results to return |

**Examples:**
```bash
# Basic search
imagerag search "dog playing in the park"

# Get more results
imagerag search "sunset" --top 25

# Search for people
imagerag search "group of friends laughing"

# Search for a specific scene
imagerag search "food on a table"

# Search for video moments
imagerag search "someone dancing" --top 20
```

**Output format:**
```
  1. /home/user/Pictures/trip.jpg      score=0.312
  2. /home/user/Pictures/beach.jpg     score=0.289
  3. /home/user/Videos/vlog.mp4  @ 1:35  score=0.271
```
Score ranges from 0 to 1 — higher means a closer match.

---

### `imagerag serve`

Launches a local web UI in your browser for a visual search experience.

```bash
imagerag serve
```

| Option | Default | Description |
|---|---|---|
| `--port` | `7860` | Port to run the web UI on |
| `--share` | off | Generate a public Gradio share link |

**Examples:**
```bash
# Launch on default port (http://localhost:7860)
imagerag serve

# Use a different port
imagerag serve --port 8080

# Share publicly (generates a temporary gradio.live URL)
imagerag serve --share
```

---

### General

```bash
# Show help for any command
imagerag --help
imagerag index --help
imagerag search --help
imagerag serve --help
```

---

## How It Works

1. **Indexing:** Each image (or video keyframe) is passed through the CLIP image encoder to produce a 512-dimensional embedding vector. All vectors are stored in a FAISS index on disk.

2. **Searching:** Your text query is passed through the CLIP text encoder. The resulting vector is compared against all indexed image vectors using cosine similarity. The closest matches are returned.

3. **Why it works offline:** The CLIP model weights (~150MB) are downloaded once on first run and cached locally by `sentence-transformers`. After that, no internet is needed.

---

## Index Location

The index is stored at `~/.imagerag/`:

```
~/.imagerag/
├── index.faiss     # FAISS vector index
└── metadata.json   # file paths and video timestamps
```

To rebuild the index (e.g. after adding new photos), just run `imagerag index` again.

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## License

MIT
