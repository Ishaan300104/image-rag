import json
import base64
import webbrowser
import tempfile
import numpy as np
import faiss
import cv2
from io import BytesIO
from pathlib import Path
from PIL import Image
from sklearn.decomposition import PCA

INDEX_DIR = Path.home() / ".imagerag"
INDEX_FILE = INDEX_DIR / "index.faiss"
META_FILE = INDEX_DIR / "metadata.json"

THUMB_SIZE = (80, 80)
THUMB_QUALITY = 65


def _load_embeddings():
    if not INDEX_FILE.exists() or not META_FILE.exists():
        raise FileNotFoundError(
            "No index found. Run `imagerag index <directory>` first."
        )
    index = faiss.read_index(str(INDEX_FILE))
    embeddings = index.reconstruct_n(0, index.ntotal)
    with open(META_FILE) as f:
        metadata = json.load(f)
    return embeddings, metadata


def _reduce_to_3d(embeddings):
    pca = PCA(n_components=3)
    coords = pca.fit_transform(embeddings)
    # Normalize to roughly -1..1 range for consistent scene scale
    scale = np.abs(coords).max()
    return (coords / scale).tolist()


def _image_to_base64(img: Image.Image) -> str:
    img = img.convert("RGB")
    img.thumbnail(THUMB_SIZE, Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=THUMB_QUALITY)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _extract_frame_at(video_path: str, timestamp: float) -> Image.Image | None:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(timestamp * fps))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def _build_thumbnail(entry: dict) -> str:
    """Return a base64 data URI for the entry, or empty string on failure."""
    try:
        if entry["type"] == "image":
            path = Path(entry["path"])
            if path.exists():
                return _image_to_base64(Image.open(path))
        elif entry["type"] == "video":
            frame = _extract_frame_at(entry["path"], entry.get("timestamp", 0))
            if frame:
                return _image_to_base64(frame)
    except Exception:
        pass
    return ""


def _make_label(entry: dict) -> str:
    path = Path(entry["path"])
    if entry["type"] == "video":
        ts = entry.get("timestamp", 0)
        m, s = divmod(int(ts), 60)
        return f"{path.name} @ {m}:{s:02d}"
    return path.name


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>ImageRAG — Embedding Visualizer</title>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ background: #0d1117; overflow: hidden; font-family: monospace; }}
    #info {{
      position: absolute; top: 14px; left: 14px; color: #8b949e; font-size: 13px;
      pointer-events: none; user-select: none;
    }}
    #info b {{ color: #e6edf3; }}
    #tooltip {{
      position: absolute; display: none;
      background: rgba(13,17,23,0.92); border: 1px solid #30363d;
      border-radius: 6px; padding: 6px 8px; color: #e6edf3;
      font-size: 11px; pointer-events: none; max-width: 220px;
      word-break: break-all;
    }}
  </style>
</head>
<body>
  <div id="info"><b>ImageRAG</b> &mdash; {n} embeddings &nbsp;|&nbsp; drag to rotate &nbsp; scroll to zoom &nbsp; right-drag to pan</div>
  <div id="tooltip"></div>

  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
  <script>
    const DATA = {data_json};

    // ── Renderer ────────────────────────────────────────────────────────────
    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.001, 100);
    camera.position.set(0, 0, 3.5);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    // ── Spoke lines from origin ──────────────────────────────────────────────
    const linePos = [];
    DATA.coords.forEach(([x, y, z]) => linePos.push(0, 0, 0, x, y, z));
    const lineGeo = new THREE.BufferGeometry();
    lineGeo.setAttribute('position', new THREE.Float32BufferAttribute(linePos, 3));
    scene.add(new THREE.LineSegments(lineGeo,
      new THREE.LineBasicMaterial({{ color: 0x1e3a5f, transparent: true, opacity: 0.5 }})));

    // Origin marker
    const originGeo = new THREE.SphereGeometry(0.012, 8, 8);
    const originMat = new THREE.MeshBasicMaterial({{ color: 0xffffff }});
    scene.add(new THREE.Mesh(originGeo, originMat));

    // ── Image sprites ────────────────────────────────────────────────────────
    const sprites = [];
    const loader = new THREE.TextureLoader();

    DATA.coords.forEach(([x, y, z], i) => {{
      const src = DATA.thumbnails[i];
      const label = DATA.labels[i];
      const isVideo = DATA.types[i] === 'video';

      const onLoad = (tex) => {{
        const mat = new THREE.SpriteMaterial({{ map: tex, sizeAttenuation: true }});
        const sprite = new THREE.Sprite(mat);
        sprite.position.set(x, y, z);
        sprite.scale.set(0.13, 0.13, 1);
        sprite.userData.label = label;
        scene.add(sprite);
        sprites.push(sprite);

        // Orange tint border for video frames
        if (isVideo) {{
          const borderTex = tex.clone();
          const bmat = new THREE.SpriteMaterial({{
            map: borderTex, color: 0xff8a65,
            transparent: true, opacity: 0.35,
          }});
          const border = new THREE.Sprite(bmat);
          border.position.set(x, y, z);
          border.scale.set(0.145, 0.145, 1);
          scene.add(border);
        }}
      }};

      if (src) {{
        loader.load(src, onLoad);
      }} else {{
        // Fallback colored square for missing thumbnails
        const canvas = document.createElement('canvas');
        canvas.width = canvas.height = 64;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = isVideo ? '#ff8a65' : '#4fc3f7';
        ctx.fillRect(0, 0, 64, 64);
        ctx.fillStyle = 'rgba(0,0,0,0.4)';
        ctx.font = '28px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(isVideo ? '🎬' : '🖼', 32, 42);
        const tex = new THREE.CanvasTexture(canvas);
        onLoad(tex);
      }}
    }});

    // ── Hover tooltip ────────────────────────────────────────────────────────
    const raycaster = new THREE.Raycaster();
    raycaster.params.Sprite = {{ threshold: 0.08 }};
    const mouse = new THREE.Vector2();
    const tooltip = document.getElementById('tooltip');

    window.addEventListener('mousemove', e => {{
      mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
      mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
      raycaster.setFromCamera(mouse, camera);
      const hits = raycaster.intersectObjects(sprites);
      if (hits.length) {{
        tooltip.style.display = 'block';
        tooltip.style.left = (e.clientX + 14) + 'px';
        tooltip.style.top = (e.clientY + 14) + 'px';
        tooltip.textContent = hits[0].object.userData.label;
      }} else {{
        tooltip.style.display = 'none';
      }}
    }});

    // ── Animate ──────────────────────────────────────────────────────────────
    (function animate() {{
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }})();

    window.addEventListener('resize', () => {{
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }});
  </script>
</body>
</html>
"""


def visualize(output_html=None):
    """
    Reduce embeddings to 3D with PCA and render an interactive Three.js scene
    where each data point is the actual image thumbnail. Opens in browser.
    """
    from rich.progress import Progress, BarColumn, TaskProgressColumn, TextColumn

    embeddings, metadata = _load_embeddings()

    print(f"Reducing {len(embeddings)} embeddings to 3D...")
    coords = _reduce_to_3d(embeddings)

    thumbnails = []
    labels = []
    types = []

    with Progress(
        TextColumn("[bold blue]Building thumbnails"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.completed}/{task.total}"),
    ) as progress:
        task = progress.add_task("", total=len(metadata))
        for entry in metadata:
            thumbnails.append(_build_thumbnail(entry))
            labels.append(_make_label(entry))
            types.append(entry["type"])
            progress.advance(task)

    data = json.dumps({"coords": coords, "thumbnails": thumbnails, "labels": labels, "types": types})
    html = HTML_TEMPLATE.format(n=len(metadata), data_json=data)

    if output_html:
        out = Path(output_html)
    else:
        out = Path(tempfile.mktemp(suffix=".html", prefix="imagerag_viz_"))

    out.write_text(html, encoding="utf-8")
    webbrowser.open(f"file://{out.resolve()}")
    print(f"Visualization saved to: {out}")
