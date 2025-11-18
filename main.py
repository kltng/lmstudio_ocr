import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import lmstudio as lms
from PIL import Image
from bs4 import BeautifulSoup

from prompts import PROMPT_MAPPING


def setup_logging():
    """Setup comprehensive logging system"""
    log_dir = Path("output/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"ocr_processing_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    return logging.getLogger(__name__)


def setup_output_directories():
    """Create output directory structure"""
    output_dirs = {
        "markdown": "output/markdown",
        "markdown_with_headers": "output/markdown_with_headers",
        "html_with_labels": "output/html_with_labels",
        "images_with_bboxes": "output/images_with_bboxes",
        "merged_markdown": "output/merged_markdown",
        "logs": "output/logs",
    }

    for dir_path in output_dirs.values():
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    return output_dirs


def get_image_files(input_dir="input"):
    """Get all supported image files from input directory"""
    input_path = Path(input_dir)
    if not input_path.exists():
        logging.error(f"Input directory {input_dir} does not exist")
        sys.exit(1)

    # Supported image formats
    extensions = ["*.png", "*.jpg", "*.jpeg", "*.tiff", "*.tif"]
    image_files = []

    for ext in extensions:
        image_files.extend(input_path.glob(ext))

    image_files.sort()  # Sort for consistent processing order
    return image_files


def needs_processing(image_path, requested_format, output_dirs):
    """Check if image needs to be processed based on existing output files"""
    base_name = image_path.stem

    # Check what outputs already exist
    existing_outputs = {
        "markdown_with_headers": Path(output_dirs["markdown_with_headers"])
        / f"{base_name}.md",
        "markdown": Path(output_dirs["markdown"]) / f"{base_name}.md",
        "html_with_labels": Path(output_dirs["html_with_labels"]) / f"{base_name}.html",
        "images_with_bboxes": Path(output_dirs["images_with_bboxes"])
        / f"{base_name}_bboxes.png",
    }

    if requested_format == "all":
        # Skip only if ALL outputs exist
        return not all(path.exists() for path in existing_outputs.values())
    else:
        # Map old format names to new ones
        format_map = {
            "markdown_with_labels": "markdown_with_headers",
            "markdown_no_labels": "markdown",
            "html": "html_with_labels",
            "images": "images_with_bboxes",
        }
        actual_format = format_map.get(requested_format)
        if actual_format is None:
            actual_format = requested_format
        output_path = existing_outputs.get(actual_format)
        if output_path is None:
            return True
        return not output_path.exists()


def get_missing_formats(image_path, requested_format, output_dirs):
    """Get list of formats that need to be generated"""
    base_name = image_path.stem

    all_formats = [
        "markdown_with_headers",
        "markdown",
        "html_with_labels",
        "images_with_bboxes",
    ]

    if requested_format == "all":
        return [
            fmt
            for fmt in all_formats
            if not (
                Path(output_dirs[fmt]) / f"{base_name}.md"
                if fmt != "images_with_bboxes"
                else Path(output_dirs[fmt]) / f"{base_name}_bboxes.png"
            ).exists()
        ]
    else:
        return (
            [requested_format]
            if needs_processing(image_path, requested_format, output_dirs)
            else []
        )


def process_image(client, image_path, prompt_type="ocr", model_name="chandra-ocr"):
    """Process a single image and return OCR result"""
    try:
        logging.info(f"Processing {image_path.name} with prompt: {prompt_type}")

        image_handle = client.files.prepare_image(str(image_path))
        model = client.llm.model(model_name)
        chat = lms.Chat()
        chat.add_user_message(PROMPT_MAPPING[prompt_type], images=[image_handle])

        start_time = time.time()
        prediction = model.respond(chat)
        processing_time = time.time() - start_time

        logging.info(f"Completed {image_path.name} in {processing_time:.2f} seconds")
        return prediction, processing_time

    except Exception as e:
        logging.error(f"Error processing {image_path}: {e}")
        return None, 0


def parse_ocr_response(prediction):
    """Parse OCR response and extract HTML content"""
    try:
        # Handle PredictionResult object
        if hasattr(prediction, "content"):
            prediction_str = prediction.content
        else:
            prediction_str = str(prediction)

        # Try to parse as JSON first
        try:
            prediction_data = json.loads(prediction_str)

            # Handle different response structures
            if isinstance(prediction_data, dict):
                # For layout OCR, look for structured HTML with data-label attributes first
                layout_html = prediction_data.get("layout_html") or prediction_data.get(
                    "structured_html"
                )
                if layout_html:
                    return layout_html

                # Look for HTML content in various fields
                html_content = (
                    prediction_data.get("html")
                    or prediction_data.get("content")
                    or prediction_data.get("text", "")
                )

                # If content is a list of paragraphs, join them
                if isinstance(html_content, list):
                    paragraphs = []
                    for item in html_content:
                        if isinstance(item, dict) and "paragraph" in item:
                            paragraphs.append(item["paragraph"])
                        elif isinstance(item, str):
                            paragraphs.append(item)
                    html_content = "\n\n".join(paragraphs)

                return html_content
            else:
                # If it's not a dict, treat as string
                return str(prediction_data)

        except json.JSONDecodeError:
            # If not JSON, treat as plain HTML
            return prediction_str

    except Exception as e:
        logging.error(f"Error parsing OCR response: {e}")
        return ""


def html_to_markdown(html_content, include_headers_footers=True):
    """Convert HTML content to markdown"""
    if not html_content:
        return ""

    try:
        soup = BeautifulSoup(html_content, "html.parser")

        # Remove headers and footers if not included
        if not include_headers_footers:
            for div in soup.find_all("div", attrs={"data-label": True}):
                label_attr = div.get("data-label")
                if label_attr:
                    label = str(label_attr).lower()
                    if label in ["page-header", "page-footer"]:
                        div.decompose()  # Remove the element entirely

        # Convert paragraphs
        for p in soup.find_all("p"):
            p.replace_with(f"\n\n{p.get_text()}")

        # Convert headers
        for i in range(1, 7):
            for header in soup.find_all(f"h{i}"):
                header.replace_with(f"\n\n{'#' * i} {header.get_text()}\n\n")

        # Convert line breaks
        for br in soup.find_all("br"):
            br.replace_with("\n")

        # Convert bold and italic
        for b in soup.find_all(["b", "strong"]):
            b.replace_with(f"**{b.get_text()}**")

        for i in soup.find_all(["i", "em"]):
            i.replace_with(f"*{i.get_text()}*")

        # Clean up extra whitespace
        text = soup.get_text()
        text = "\n".join(line.strip() for line in text.split("\n") if line.strip())

        return text.strip()

    except Exception as e:
        logging.error(f"Error converting HTML to markdown: {e}")
        return html_content


def save_markdown_with_labels(html_content, output_path, base_name):
    """Save markdown with headers and footers included"""
    markdown_content = html_to_markdown(html_content, include_headers_footers=True)
    output_file = output_path / f"{base_name}.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    logging.info(f"Saved markdown with labels: {output_file}")


def save_markdown_no_labels(html_content, output_path, base_name):
    """Save markdown without headers and footers"""
    markdown_content = html_to_markdown(html_content, include_headers_footers=False)
    output_file = output_path / f"{base_name}.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    logging.info(f"Saved markdown without labels: {output_file}")


def save_html_with_labels(html_content, output_path, base_name):
    """Save HTML with proper structure"""
    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>OCR Result - {base_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1, h2, h3, h4, h5, h6 {{ color: #333; }}
        p {{ margin-bottom: 1em; }}
        table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
{html_content}
</body>
</html>"""
    output_file = output_path / f"{base_name}.html"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(full_html)
    logging.info(f"Saved HTML with labels: {output_file}")


def save_images_with_bboxes(html_content, image_path, output_path, base_name):
    """Save image with bounding boxes (placeholder for now)"""
    try:
        # For now, just copy the original image
        # TODO: Implement actual bbox drawing when layout data is available
        image = Image.open(image_path)
        output_file = output_path / f"{base_name}_bboxes.png"
        image.save(output_file)
        logging.info(f"Saved image with bboxes: {output_file}")
    except Exception as e:
        logging.error(f"Error saving image for {base_name}: {e}")


def process_output_formats(html_content, image_path, formats_to_process, output_dirs):
    """Process and save outputs for specified formats"""
    base_name = image_path.stem

    for format_type in formats_to_process:
        try:
            if format_type == "markdown_with_headers":
                save_markdown_with_labels(
                    html_content, Path(output_dirs[format_type]), base_name
                )
            elif format_type == "markdown":
                save_markdown_no_labels(
                    html_content, Path(output_dirs[format_type]), base_name
                )
            elif format_type == "html_with_labels":
                save_html_with_labels(
                    html_content, Path(output_dirs[format_type]), base_name
                )
            elif format_type == "images_with_bboxes":
                save_images_with_bboxes(
                    html_content, image_path, Path(output_dirs[format_type]), base_name
                )
        except Exception as e:
            logging.error(f"Error processing format {format_type} for {base_name}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="LM Studio OCR - Enhanced batch processing with multiple output formats"
    )
    parser.add_argument(
        "--format",
        choices=["markdown_with_headers", "markdown", "html", "images", "all"],
        default="all",
        help="Output format (default: all)",
    )
    parser.add_argument(
        "--input-dir",
        default="input",
        help="Input directory containing images (default: input)",
    )
    parser.add_argument(
        "--model",
        default="chandra-ocr",
        help="LM Studio model name (default: chandra-ocr)",
    )

    args = parser.parse_args()

    # Setup logging and directories
    logger = setup_logging()
    output_dirs = setup_output_directories()

    # Map format arguments to internal format names
    format_mapping = {
        "markdown_with_headers": "markdown_with_headers",
        "markdown": "markdown",
        "html": "html_with_labels",
        "images": "images_with_bboxes",
        "all": "all",
    }

    requested_format = format_mapping[args.format]

    # Get all image files
    image_files = get_image_files(args.input_dir)

    if not image_files:
        logger.error(f"No supported image files found in {args.input_dir}")
        return

    logger.info(f"Found {len(image_files)} images to process")
    logger.info(f"Output format: {args.format}")
    logger.info(f"Using model: {args.model}")
    logger.info("-" * 60)

    # Process images one by one
    processed_count = 0
    skipped_count = 0
    total_processing_time = 0

    with lms.Client() as client:
        for i, image_path in enumerate(image_files, 1):
            # Check if processing is needed
            if not needs_processing(image_path, requested_format, output_dirs):
                logger.info(f"Skipping {image_path.name} (already processed)")
                skipped_count += 1
                continue

            # Get missing formats
            formats_to_process = get_missing_formats(
                image_path, requested_format, output_dirs
            )
            if not formats_to_process:
                logger.info(f"Skipping {image_path.name} (all requested formats exist)")
                skipped_count += 1
                continue

            logger.info(f"[{i}/{len(image_files)}] Processing {image_path.name}")

            # Choose prompt based on whether we need layout information
            needs_layout = (
                "images_with_bboxes" in formats_to_process
                or "markdown_with_headers" in formats_to_process
                or "markdown" in formats_to_process
            )
            prompt_type = "ocr_layout" if needs_layout else "ocr"

            # Process image
            prediction, processing_time = process_image(
                client, image_path, prompt_type, args.model
            )

            if prediction:
                html_content = parse_ocr_response(prediction)
                if html_content:
                    process_output_formats(
                        html_content, image_path, formats_to_process, output_dirs
                    )
                    processed_count += 1
                    total_processing_time += processing_time
                else:
                    logger.error(f"No content extracted from {image_path.name}")
            else:
                logger.error(f"Failed to process {image_path.name}")

    # Final summary
    logger.info("-" * 60)
    logger.info("Processing Summary:")
    logger.info(f"Total images found: {len(image_files)}")
    logger.info(f"Images processed: {processed_count}")
    logger.info(f"Images skipped: {skipped_count}")
    logger.info(f"Total processing time: {total_processing_time:.2f} seconds")
    if processed_count > 0:
        logger.info(
            f"Average time per image: {total_processing_time / processed_count:.2f} seconds"
        )
    logger.info("Processing complete!")


if __name__ == "__main__":
    main()
