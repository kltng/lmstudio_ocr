# LM Studio OCR - Enhanced Batch Processing Tool

A powerful OCR tool that processes images using LM Studio's Chandra OCR model with multiple output formats and smart progress tracking.

## Features

- 🚀 **Batch Processing**: Process multiple images sequentially
- 📁 **Smart Resume**: Automatically skips already processed images
- 📝 **Multiple Output Formats**: Markdown, HTML, and annotated images
- 📊 **Comprehensive Logging**: Detailed processing logs with timestamps
- 🌐 **Multi-format Support**: PNG, JPEG, JPG, TIFF, TIF files
- 🎯 **One-by-One Processing**: Saves results immediately after each image
- 🔄 **Partial Processing**: Add new formats without reprocessing everything

## Installation

```bash
# Install dependencies
uv sync

# Verify installation
uv run python main.py --help
```

## Usage

### Basic Usage

```bash
# Process all formats for all images
uv run python main.py --format all --input-dir input_images

# Process specific format only
uv run python main.py --format html --input-dir input_images
uv run python main.py --format markdown_with_headers --input-dir input_images
uv run python main.py --format markdown --input-dir input_images
uv run python main.py --format images --input-dir input_images
```

### CLI Options

| Option | Description | Default |
|---------|-------------|----------|
| `--format` | Output format to generate (markdown_with_headers, markdown, html, images, all) | `all` |
| `--input-dir` | Input directory containing images | `input_images` |
| `--model` | LM Studio model name | `chandra-ocr` |

### Output Formats

| Format | Description | Output File |
|--------|-------------|--------------|
| `markdown_with_headers` | Markdown with page headers and footers | `{filename}.md` |
| `markdown` | Markdown without page headers/footers | `{filename}.md` |
| `html` | Styled HTML with proper structure | `{filename}.html` |
| `images` | Images with bounding boxes | `{filename}_bboxes.png` |
| `all` | Generate all formats | All above files |

### Output Structure

```
output/
├── logs/                           # Processing logs with timestamps
│   └── ocr_processing_YYYYMMDD_HHMMSS.log
├── markdown_with_headers/             # Markdown output with headers/footers
│   └── image1.md
├── markdown/                         # Markdown output without headers/footers  
│   └── image1.md
├── html_with_labels/               # HTML output with styling
│   └── image1.html
└── images_with_bboxes/             # Images with bounding boxes
    └── image1_bboxes.png
```

### Merging Markdown Files

After processing, you can merge all markdown files into a single document:

```bash
# Merge markdown with headers
uv run python merge_markdown.py output/markdown_with_headers merged_with_headers.md

# Merge markdown without headers  
uv run python merge_markdown.py output/markdown merged_without_headers.md

# Merge all markdown files (sorted by filename)
uv run python merge_markdown.py output/markdown all_merged.md
```

## Examples

### Process All Images with All Formats
```bash
uv run python main.py --format all --input-dir input_images
```

### Generate Only HTML Output
```bash
uv run python main.py --format html --input-dir input_images
```

### Process Different Input Directory
```bash
uv run python main.py --format all --input-dir /path/to/images
```

### Use Different Model
```bash
uv run python main.py --format all --model custom-ocr-model
```

## Smart Features

### Resume Capability
The script automatically detects already processed files and skips them:
- Checks each output folder for existing files
- Only processes missing formats
- Can interrupt and resume anytime

### Progress Tracking
- Real-time progress: `[1/6] Processing image.png`
- Processing time per image
- Summary report with total time and averages

### Error Handling
- Continues processing even if some images fail
- Detailed error logging
- Graceful handling of corrupted images

### Logging
All processing activities are logged to `output/logs/` with:
- Timestamp for each operation
- Success/failure status
- Processing time metrics
- Error details and stack traces

## Supported Image Formats

- PNG (`.png`)
- JPEG (`.jpg`, `.jpeg`)
- TIFF (`.tiff`, `.tif`)

## Development

### Code Quality
```bash
# Lint code
uv run ruff check .

# Format code
uv run ruff format .

# Type checking
uv run mypy .
```

### Testing
```bash
# Run tests (when available)
uv run pytest
```

## Requirements

- Python 3.13+
- LM Studio running with Chandra OCR model
- UV package manager

## Output Quality

The tool uses official Chandra OCR prompts for:
- ✅ High accuracy text recognition
- ✅ Proper layout preservation
- ✅ Structured HTML output
- ✅ Bounding box information
- ✅ Multi-language support
- ✅ Table and form handling

## Troubleshooting

### Common Issues

1. **LM Studio Not Running**
   - Ensure LM Studio is running locally
   - Check that Chandra OCR model is loaded

2. **No Images Found**
   - Verify input directory exists
   - Check image file extensions (PNG, JPG, JPEG, TIFF)

3. **Permission Errors**
   - Ensure write access to output directory
   - Check file permissions for input images

4. **Processing Timeouts**
   - Large images may take longer to process
   - Check LM Studio resource usage

### Log Analysis
Check detailed logs in `output/logs/` for:
- Processing errors
- Performance metrics
- File-specific issues

## License

This tool uses Chandra OCR by Datalab (https://github.com/datalab-to/chandra) and follows their OCR guidelines.