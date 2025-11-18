import time
import subprocess
import tempfile
import shutil
import logging
from pathlib import Path
from typing import Generator, List, Optional, Any, Tuple

import gradio as gr

FORMATS = [
    "all",
    "markdown_with_headers",
    "markdown",
    "html",
    "images",
]


def _extract_dir_path(output_dir: Optional[str]) -> str:
    """Extract directory path from text input."""
    if not output_dir or not output_dir.strip():
        return "output"
    return output_dir.strip()


def _read_latest_log(output_dir: str = "output", max_chars: int = 6000) -> str:
    """Read the tail of the newest log file if available."""
    try:
        logs_dir = Path(output_dir) / "logs"
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


def _collect_outputs(output_dir: str = "output") -> Tuple[List[str], List[str]]:
    """Return (gallery_images, downloadable_files)."""
    out_root = Path(output_dir)
    if not out_root.exists():
        return [], []
    
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
    output_dir: Optional[str],
    files: Optional[List[Any]] = None,
    progress: gr.Progress = gr.Progress(),
) -> Generator[Tuple[str, List[str], List[str]], None, None]:
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

    # Extract output directory path from file picker result
    effective_output = _extract_dir_path(output_dir)

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
    
    # Note: main.py currently uses hardcoded "output" directory
    # The output_dir parameter is stored for merge_markdown function
    # TODO: Add --output-dir parameter to main.py for full support

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
            live_tail = _read_latest_log(effective_output)
            if live_tail:
                logs_accum = live_tail
            # Collect outputs so far (best-effort)
            gallery_imgs, download_files = _collect_outputs(effective_output)
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

    tail = _read_latest_log(effective_output)
    logs_accum = tail or (stdout or "")
    if stderr:
        logs_accum += "\n\n[stderr]\n" + stderr[-2000:]

    gallery_imgs, download_files = _collect_outputs(effective_output)

    # Cleanup temporary upload dir
    if work_dir and work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)

    progress(1.0, desc="Done")
    yield logs_accum[-6000:], gallery_imgs, download_files


def merge_markdown_ui(
    markdown_type: str,
    output_dir: Optional[str],
    output_filename: str,
    progress: gr.Progress = gr.Progress(),
) -> Tuple[str, str]:
    """Merge markdown files from the specified directory."""
    try:
        progress(0.1, desc="Finding markdown files...")
        
        # Extract output directory path from file picker result
        out_path = _extract_dir_path(output_dir)
        
        # Determine which markdown directory to use
        out_root = Path(out_path)
        if markdown_type == "with_headers":
            md_dir = out_root / "markdown_with_headers"
        else:
            md_dir = out_root / "markdown"
        
        if not md_dir.exists():
            return f"Error: Directory {md_dir} does not exist", ""
        
        # Find all markdown files
        md_files = list(md_dir.glob("*.md"))
        
        if not md_files:
            return f"No markdown files found in {md_dir}", ""
        
        # Sort files by filename
        md_files.sort(key=lambda x: x.name)
        
        progress(0.3, desc=f"Merging {len(md_files)} files...")
        
        # Create merged content
        merged_content = []
        merged_content.append("# Merged OCR Results\n")
        merged_content.append(f"*Generated from {len(md_files)} files*\n")
        merged_content.append("---\n\n")
        
        for i, md_file in enumerate(md_files, 1):
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                
                if content:
                    filename = md_file.stem
                    merged_content.append(f"## {filename}\n\n")
                    merged_content.append(content)
                    merged_content.append("\n---\n\n")
            except Exception as e:
                logging.warning(f"Error reading {md_file}: {e}")
        
        progress(0.8, desc="Writing merged file...")
        
        # Create output directory
        merged_output_dir = out_root / "merged_markdown"
        merged_output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = merged_output_dir / (output_filename or "merged_markdown.md")
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("".join(merged_content))
        
        progress(1.0, desc="Done")
        return (
            f"Successfully merged {len(md_files)} files into {output_file}",
            str(output_file),
        )
    except Exception as e:
        return f"Error merging markdown: {e}", ""


with gr.Blocks(title="LM Studio OCR UI") as demo:
    gr.Markdown("## LM Studio OCR — Batch Processor")

    with gr.Row():
        fmt = gr.Dropdown(FORMATS, value="all", label="Output format")
        model = gr.Textbox(
            value="chandra-ocr",
            label="Model (any LM Studio vision model)",
            info="Examples: chandra-ocr, qwen2.5-vl-7b, gemma-3-12b, internvl-1.5",
        )
    
    with gr.Row():
        output_dir = gr.Textbox(
            value="output",
            label="Output folder",
            placeholder="Enter folder path (e.g., output, /path/to/folder)",
            info="Directory where processed files will be saved",
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

    def toggle_rows(choice: str) -> Tuple[gr.update, gr.update]:
        return gr.update(visible=(choice == "Use folder")), gr.update(
            visible=(choice == "Upload images")
        )

    source.change(toggle_rows, source, [folder_row, upload_row])

    run_btn.click(
        run_ocr,
        inputs=[fmt, model, source, input_dir, output_dir, files],
        outputs=[logs, gallery, downloads],
    )
    
    # Merge Markdown Section
    gr.Markdown("---")
    gr.Markdown("## Merge Markdown Files")
    
    with gr.Row():
        merge_md_type = gr.Radio(
            choices=["with_headers", "without_headers"],
            value="without_headers",
            label="Markdown type",
            info="Merge markdown with or without headers/footers",
        )
        merge_output_filename = gr.Textbox(
            value="merged_markdown.md",
            label="Output filename",
            info="Name for the merged markdown file",
        )
    
    merge_btn = gr.Button("Merge Markdown", variant="secondary")
    merge_status = gr.Textbox(label="Merge Status", lines=2)
    merge_download = gr.File(label="Download Merged Markdown")
    
    merge_btn.click(
        merge_markdown_ui,
        inputs=[merge_md_type, output_dir, merge_output_filename],
        outputs=[merge_status, merge_download],
    )


# Default host/port for convenience
if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860)
