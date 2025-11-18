import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple

import lmstudio as lms
from PIL import Image, ImageDraw, ImageFont
from bs4 import BeautifulSoup

from prompts import PROMPT_MAPPING

# Format name mapping: maps CLI/old format names to internal format names
FORMAT_MAPPING = {
    "markdown_with_labels": "markdown_with_headers",
    "markdown_no_labels": "markdown",
    "html": "html_with_labels",
    "images": "images_with_bboxes",
    # Internal format names map to themselves
    "markdown_with_headers": "markdown_with_headers",
    "markdown": "markdown",
    "html_with_labels": "html_with_labels",
    "images_with_bboxes": "images_with_bboxes",
    "all": "all",
}


def normalize_format_name(format_name: str) -> str:
    """Normalize format name to internal format name"""
    return FORMAT_MAPPING.get(format_name, format_name)


def setup_logging() -> logging.Logger:
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


def setup_output_directories() -> dict[str, str]:
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


def get_image_files(input_dir: str = "input") -> list[Path]:
    """Get all supported image files from input directory"""
    input_path = Path(input_dir)
    if not input_path.exists():
        logging.error(f"Input directory {input_dir} does not exist")
        sys.exit(1)

    # Supported image formats
    extensions = ["*.png", "*.jpg", "*.jpeg", "*.tiff", "*.tif"]
    image_files: list[Path] = []

    for ext in extensions:
        image_files.extend(input_path.glob(ext))

    image_files.sort()  # Sort for consistent processing order
    return image_files


def needs_processing(image_path: Path, requested_format: str, output_dirs: dict[str, str]) -> bool:
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
        # Normalize format name to internal format
        actual_format = normalize_format_name(requested_format)
        output_path = existing_outputs.get(actual_format)
        if output_path is None:
            return True
        return not output_path.exists()


def get_missing_formats(image_path: Path, requested_format: str, output_dirs: dict[str, str]) -> list[str]:
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
        # Normalize format name to internal format before returning
        normalized_format = normalize_format_name(requested_format)
        return (
            [normalized_format]
            if needs_processing(image_path, requested_format, output_dirs)
            else []
        )


def process_image(
    client: lms.Client, image_path: Path, prompt_type: str = "ocr", model_name: str = "chandra-ocr"
) -> Tuple[Optional[Any], float]:
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

    except (IOError, OSError) as e:
        logging.error(f"Error reading image file {image_path}: {e}")
        return None, 0.0
    except KeyError as e:
        logging.error(f"Invalid prompt type '{prompt_type}' or model '{model_name}' not found: {e}")
        return None, 0.0
    except Exception as e:
        logging.error(f"Unexpected error processing {image_path} with model {model_name}: {e}")
        return None, 0.0


def parse_ocr_response(prediction: Any) -> str:
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

    except (AttributeError, TypeError) as e:
        logging.error(f"Error accessing prediction content: {e}")
        return ""
    except Exception as e:
        logging.error(f"Unexpected error parsing OCR response: {e}")
        return ""


def html_to_markdown(html_content: str, include_headers_footers: bool = True) -> str:
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

    except (AttributeError, TypeError) as e:
        logging.error(f"Error parsing HTML structure: {e}")
        return html_content
    except Exception as e:
        logging.error(f"Unexpected error converting HTML to markdown: {e}")
        return html_content


def save_markdown_with_labels(html_content: str, output_path: Path, base_name: str) -> None:
    """Save markdown with headers and footers included"""
    markdown_content = html_to_markdown(html_content, include_headers_footers=True)
    output_file = output_path / f"{base_name}.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    logging.info(f"Saved markdown with labels: {output_file}")


def save_markdown_no_labels(html_content: str, output_path: Path, base_name: str) -> None:
    """Save markdown without headers and footers"""
    markdown_content = html_to_markdown(html_content, include_headers_footers=False)
    output_file = output_path / f"{base_name}.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    logging.info(f"Saved markdown without labels: {output_file}")


def save_html_with_labels(html_content: str, output_path: Path, base_name: str) -> None:
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


def save_images_with_bboxes(html_content: str, image_path: Path, output_path: Path, base_name: str) -> None:
    """Save image with bounding boxes drawn from HTML data-bbox attributes.
    
    Parses HTML to extract div elements with data-bbox attributes, scales coordinates
    from normalized 0-1024 range to image dimensions, and draws rectangles with labels.
    """
    try:
        # Open the original image
        image = Image.open(image_path)
        img_width, img_height = image.size
        
        # Create a copy for drawing
        img_with_bboxes = image.copy()
        draw = ImageDraw.Draw(img_with_bboxes)
        
        # Try to load a font for labels (fallback to default if not available)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        except (OSError, IOError):
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except (OSError, IOError):
                font = ImageFont.load_default()
        
        # Parse HTML to find divs with data-bbox attributes
        soup = BeautifulSoup(html_content, "html.parser")
        bbox_count = 0
        
        # Find all divs with data-bbox attribute
        for div in soup.find_all("div", attrs={"data-bbox": True}):
            bbox_attr = div.get("data-bbox")
            label_attr = div.get("data-label", "")
            
            if not bbox_attr:
                continue
            
            # Parse bbox coordinates: format is [x0, y0, x1, y1] normalized 0-1024
            # Handle both string format "[x0, y0, x1, y1]" and list format
            try:
                # Try to parse as JSON array first
                if isinstance(bbox_attr, str):
                    # Remove brackets and split by comma
                    coords_str = bbox_attr.strip("[]")
                    coords = [float(x.strip()) for x in coords_str.split(",")]
                else:
                    coords = list(bbox_attr)
                
                if len(coords) != 4:
                    logging.warning(f"Invalid bbox format for {base_name}: {bbox_attr}")
                    continue
                
                x0_norm, y0_norm, x1_norm, y1_norm = coords
                
                # Scale from normalized 0-1024 to actual image dimensions
                x0 = int((x0_norm / 1024.0) * img_width)
                y0 = int((y0_norm / 1024.0) * img_height)
                x1 = int((x1_norm / 1024.0) * img_width)
                y1 = int((y1_norm / 1024.0) * img_height)
                
                # Ensure coordinates are within image bounds
                x0 = max(0, min(x0, img_width))
                y0 = max(0, min(y0, img_height))
                x1 = max(0, min(x1, img_width))
                y1 = max(0, min(y1, img_height))
                
                # Choose color based on label type
                label_lower = label_attr.lower() if label_attr else ""
                if "header" in label_lower or "title" in label_lower:
                    color = (255, 0, 0)  # Red for headers
                elif "footer" in label_lower:
                    color = (0, 0, 255)  # Blue for footers
                elif "table" in label_lower:
                    color = (0, 255, 0)  # Green for tables
                elif "image" in label_lower or "figure" in label_lower:
                    color = (255, 165, 0)  # Orange for images
                else:
                    color = (0, 255, 255)  # Cyan for text and others
                
                # Draw rectangle (outline only, 2px width)
                draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
                
                # Draw label text above the box if there's space
                if label_attr and y0 > 15:
                    try:
                        # Get text bounding box for background
                        bbox_text = draw.textbbox((x0, y0 - 15), label_attr, font=font)
                        # Draw semi-transparent background (PIL doesn't support alpha in fill, so use a darker color)
                        draw.rectangle(bbox_text, fill=(0, 0, 0))
                        # Draw text
                        draw.text((x0, y0 - 15), label_attr, fill=color, font=font)
                    except Exception:
                        # Fallback if text rendering fails - just draw text without background
                        try:
                            draw.text((x0, y0 - 15), label_attr, fill=color, font=font)
                        except Exception:
                            pass
                
                bbox_count += 1
                
            except (ValueError, TypeError, IndexError) as e:
                logging.warning(f"Error parsing bbox '{bbox_attr}' for {base_name}: {e}")
                continue
        
        # Save the image with bounding boxes
        output_file = output_path / f"{base_name}_bboxes.png"
        img_with_bboxes.save(output_file)
        
        if bbox_count > 0:
            logging.info(f"Saved image with {bbox_count} bounding boxes: {output_file}")
        else:
            logging.warning(f"No bounding boxes found in HTML for {base_name}, saved original image")
            
    except (IOError, OSError) as e:
        logging.error(f"Error reading or writing image file for {base_name}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error saving image for {base_name}: {e}")


def process_output_formats(
    html_content: str, image_path: Path, formats_to_process: list[str], output_dirs: dict[str, str]
) -> None:
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
        except (IOError, OSError) as e:
            logging.error(f"File I/O error processing format {format_type} for {base_name}: {e}")
        except KeyError as e:
            logging.error(f"Invalid format type '{format_type}' or missing output directory: {e}")
        except Exception as e:
            logging.error(f"Unexpected error processing format {format_type} for {base_name}: {e}")


def main() -> None:
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
    requested_format = normalize_format_name(args.format)

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

    # Validate LM Studio connection and model availability, then process images
    try:
        with lms.Client() as client:
            # Test connection by trying to list models
            try:
                models = client.llm.list_models()
                available_models = [m.name for m in models] if hasattr(models, "__iter__") else []
                
                # Check if requested model is available
                if args.model not in available_models:
                    logger.warning(
                        f"Model '{args.model}' not found in available models. "
                        f"Available models: {', '.join(available_models) if available_models else 'none'}. "
                        f"Attempting to use model anyway..."
                    )
                else:
                    logger.info(f"Model '{args.model}' is available and ready")
            except Exception as e:
                logger.warning(
                    f"Could not verify model availability: {e}. "
                    f"Attempting to proceed with model '{args.model}'..."
                )
            
            # Test model access by trying to get it
            try:
                test_model = client.llm.model(args.model)
                logger.info(f"Successfully connected to LM Studio and accessed model '{args.model}'")
            except Exception as e:
                logger.error(
                    f"Failed to access model '{args.model}': {e}. "
                    f"Please ensure LM Studio is running and the model is loaded."
                )
                return
            
            # Process images one by one
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
    
    except (ConnectionError, OSError) as e:
        logger.error(
            f"Failed to connect to LM Studio: {e}. "
            f"Please ensure LM Studio is running and the API server is active."
        )
        return
    except Exception as e:
        logger.error(f"Unexpected error connecting to LM Studio: {e}")
        return

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
