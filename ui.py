import time
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Any, Tuple

import gradio as gr

FORMATS = [
    "all",
    "markdown_with_headers",
    "markdown",
    "html",
    "images",
]


def _read_latest_log(max_chars: int = 6000) -> str:
    """Read the tail of the newest log file if available."""
    try:
        logs_dir = Path("output/logs")
        if not logs_dir.exists():
            return ""
        log_files = sorted(logs_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not log_files:
            return ""
        latest = log_files[0]
        text = latest.read_text(encoding="utf-8", errors="ignore")
        return text[-max_chars:]
    except Exception:
        return ""


def _collect_outputs() -> Tuple[List[str], List[str]]:
    """Return (gallery_images, downloadable_files)."""
    out_root = Path("output")
    html_files = list((out_root / "html_with_labels").glob("*.html"))
    md_files = list((out_root / "markdown").glob("*.md")) + list(
        (out_root / "markdown_with_headers").glob("*.md")
    )
    img_files = list((out_root / "images_with_bboxes").glob("*_bboxes.png"))

    gallery_imgs = [str(p) for p in img_files]
    files_for_download = [str(p) for p in html_files + md_files]
    return gallery_imgs, files_for_download


def run_ocr(
    fmt: str,
    model: str,
    input_source: str,
    input_dir: str,
    files: Optional[List[Any]] = None,
    progress=gr.Progress(),
):
    """Run OCR as a streaming task, yielding live logs and outputs."""
    progress(0.02, desc="Initializing")

    # Prepare input directory
    work_dir: Optional[Path] = None
    if input_source == "Upload images":
        tmp = tempfile.mkdtemp(prefix="ocr_input_")
        work_dir = Path(tmp)
        for f in files or []:
            try:
                src = getattr(f, "name", None) or str(f)
                if not src:
                    continue
                shutil.copy(src, work_dir / Path(src).name)
            except Exception:
                continue
        effective_input = str(work_dir)
    else:
        effective_input = input_dir or "input"

    progress(0.1, desc="Starting OCR")

    cmd = [
        "uv",
        "run",
        "python",
        "main.py",
        "--format",
        fmt,
        "--input-dir",
        effective_input,
        "--model",
        model,
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError:
        # Fallback if uv not found in PATH
        proc = subprocess.Popen(
            [
                "python",
                "main.py",
                "--format",
                fmt,
                "--input-dir",
                effective_input,
                "--model",
                model,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    # Stream updates while process runs
    logs_accum = ""
    gallery_imgs: List[str] = []
    download_files: List[str] = []

    last_update = time.time()
    while True:
        ret = proc.poll()

        # Periodically update from the newest logfile and available outputs
        now = time.time()
        if now - last_update > 0.5:
            live_tail = _read_latest_log()
            if live_tail:
                logs_accum = live_tail
            # Collect outputs so far (best-effort)
            gallery_imgs, download_files = _collect_outputs()
            progress(0.2, desc="Processing…")
            yield logs_accum, gallery_imgs, download_files
            last_update = now

        if ret is not None:
            break
        time.sleep(0.1)

    # Finalize
    try:
        stdout, stderr = proc.communicate(timeout=1)
    except Exception:
        stdout, stderr = "", ""

    tail = _read_latest_log()
    logs_accum = tail or (stdout or "")
    if stderr:
        logs_accum += "\n\n[stderr]\n" + stderr[-2000:]

    gallery_imgs, download_files = _collect_outputs()

    # Cleanup temporary upload dir
    if work_dir and work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)

    progress(1.0, desc="Done")
    yield logs_accum[-6000:], gallery_imgs, download_files


with gr.Blocks(title="LM Studio OCR UI") as demo:
    gr.Markdown("## LM Studio OCR — Batch Processor")

    with gr.Row():
        fmt = gr.Dropdown(FORMATS, value="all", label="Output format")
        model = gr.Textbox(
            value="chandra-ocr",
            label="Model (any LM Studio vision model)",
            info="Examples: chandra-ocr, qwen2.5-vl-7b, gemma-3-12b, internvl-1.5",
        )

    source = gr.Radio(
        choices=["Use folder", "Upload images"],
        value="Use folder",
        label="Input source",
    )

    with gr.Row(visible=True) as folder_row:
        input_dir = gr.Textbox(value="input", label="Input folder path")

    with gr.Row(visible=False) as upload_row:
        files = gr.File(
            file_types=[".png", ".jpg", ".jpeg", ".tiff", ".tif"],
            file_count="multiple",
            label="Upload images",
        )

    run_btn = gr.Button("Run OCR", variant="primary")
    logs = gr.Textbox(label="Logs", lines=12)
    gallery = gr.Gallery(
        label="Preview images (with bboxes)", columns=3, height=300
    )
    downloads = gr.Files(label="Download outputs (HTML/Markdown)")

    def toggle_rows(choice: str):
        return gr.update(visible=(choice == "Use folder")), gr.update(
            visible=(choice == "Upload images")
        )

    source.change(toggle_rows, source, [folder_row, upload_row])

    run_btn.click(
        run_ocr,
        inputs=[fmt, model, source, input_dir, files],
        outputs=[logs, gallery, downloads],
    )


# Default host/port for convenience
if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860)
