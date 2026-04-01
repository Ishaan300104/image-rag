from pathlib import Path
from PIL import Image
import gradio as gr


def _run_search(query, top_k):
    from imagerag.searchtool import search

    if not query.strip():
        return [], "Enter a search query."

    try:
        results = search(query, top_k=int(top_k))
    except FileNotFoundError as e:
        return [], str(e)

    gallery_items = []
    text_lines = []

    for i, r in enumerate(results, 1):
        path = Path(r["path"])
        score = r["score"]

        if r["type"] == "image" and path.exists():
            img = Image.open(path)
            gallery_items.append((img, f"{path.name}  ({score:.2f})"))
            text_lines.append(f"{i}. [image] {path}  score={score:.3f}")

        elif r["type"] == "video":
            ts = r.get("timestamp", 0)
            m, s = divmod(int(ts), 60)
            text_lines.append(f"{i}. [video] {path}  @ {m}:{s:02d}  score={score:.3f}")

    summary = "\n".join(text_lines) if text_lines else "No results."
    return gallery_items, summary


def launch(port=7860, share=False):
    with gr.Blocks(title="ImageRAG", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# ImageRAG\nSearch your local photos and videos using text.")

        with gr.Row():
            query_box = gr.Textbox(
                label="Search query",
                placeholder="e.g. sunset at beach, birthday party, dog playing",
                scale=4,
            )
            top_k = gr.Slider(1, 50, value=10, step=1, label="Max results", scale=1)

        search_btn = gr.Button("Search", variant="primary")

        gallery = gr.Gallery(label="Image results", columns=4, height=500, object_fit="cover")
        text_out = gr.Textbox(label="All matches (including videos)", lines=6, interactive=False)

        search_btn.click(_run_search, inputs=[query_box, top_k], outputs=[gallery, text_out])
        query_box.submit(_run_search, inputs=[query_box, top_k], outputs=[gallery, text_out])

    demo.launch(server_port=port, share=share)
