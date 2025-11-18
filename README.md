# LM Studio OCR - Enhanced Batch Processing Tool

A powerful OCR tool that processes images using any vision-language model available in LM Studio with multiple output formats and smart progress tracking.

## Features

- 🚀 **Batch Processing**: Process multiple images sequentially
- 📁 **Smart Resume**: Automatically skips already processed images
- 📝 **Multiple Output Formats**: Markdown, HTML, and annotated images
- 📊 **Comprehensive Logging**: Detailed processing logs with timestamps
- 🌐 **Multi-format Support**: PNG, JPEG, JPG, TIFF, TIF files
- 🎯 **One-by-One Processing**: Saves results immediately after each image
- 🔄 **Partial Processing**: Add new formats without reprocessing everything

## Prerequisites

Before you begin, ensure you have the following installed:

### 1. Python 3.13+
This project requires Python 3.13 or higher. Check your version:
```bash
python --version
# or
python3 --version
```

### 2. UV Package Manager
UV is an extremely fast Python package and project manager. Install it based on your operating system:

#### macOS
```bash
# Using the official installer (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Alternative: Install via pip
pip install uv
```

#### Windows
```powershell
# Using PowerShell (recommended)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Alternative: Install via pip
pip install uv
```

#### Linux
```bash
# Using the official installer (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Alternative: Install via pip
pip install uv
```

#### Verify UV Installation
```bash
uv --version
```

### 3. Git
Git is required for cloning the repository. Install it if not already present:

#### macOS
```bash
# Install via Homebrew
brew install git

# Or download from https://git-scm.com/download/mac
```

#### Windows
Download and install from https://git-scm.com/download/win

#### Linux
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install git

# CentOS/RHEL
sudo yum install git

# Fedora
sudo dnf install git
```

#### Verify Git Installation
```bash
git --version
```

### 4. LM Studio
- Download and install LM Studio from https://lmstudio.ai/
- Install a compatible vision-language model (see recommended models below)
- Ensure LM Studio is running before using this tool

#### Recommended OCR Models

The script works with any vision-language model in LM Studio. Here are some recommended options:

**Specialized OCR Models:**
- **Chandra OCR** - Purpose-built for OCR tasks with excellent text recognition
- **Nougat** - Academic document OCR specialized for scientific papers
- **PaddleOCR-VL** - Baidu's vision-language model for OCR tasks

**General Vision-Language Models (Good for OCR):**
- **Qwen2.5-VL** (3B, 7B, 32B, 72B) - Excellent text recognition and multilingual support
- **Gemma-3** (4B, 12B, 27B) - Google's vision model with strong OCR capabilities
- **InternVL 1.5** - Fast and effective for OCR tasks
- **Phi-Vision** - Powerful but slower, good for complex documents

**How to Choose:**
- **For best OCR accuracy**: Use specialized OCR models like Chandra OCR
- **For multilingual documents**: Qwen2.5-VL or Gemma-3
- **For speed**: InternVL 1.5 or smaller Qwen2.5-VL models
- **For academic papers**: Nougat or Chandra OCR

**Model Installation:**
1. Open LM Studio
2. Search for the model name in the model browser
3. Download and load the model
4. Use the exact model name in the `--model` parameter

## Getting Started

### Step 1: Clone the Repository

Follow these steps to clone the repository to your local machine:

1. **Navigate to the repository** on GitHub
2. **Click the green "Code" button** above the file list
3. **Copy the repository URL**:
   - For HTTPS: `https://github.com/YOUR-USERNAME/lmstudio_ocr.git`
   - For SSH: `git@github.com:YOUR-USERNAME/lmstudio_ocr.git` (requires SSH key setup)
4. **Open your terminal** or command prompt
5. **Navigate to your desired directory**:
   ```bash
   cd /path/to/your/projects
   ```
6. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR-USERNAME/lmstudio_ocr.git
   ```
7. **Navigate into the cloned directory**:
   ```bash
   cd lmstudio_ocr
   ```

#### Alternative Cloning Methods

**Using GitHub CLI:**
```bash
gh repo clone YOUR-USERNAME/lmstudio_ocr
```

**Using GitHub Desktop:**
1. Click "Code" button
2. Select "Open with GitHub Desktop"
3. Follow the prompts in GitHub Desktop

### Step 2: Install Dependencies

Once you have cloned the repository, install the required dependencies:

```bash
# Install all project dependencies
uv sync

# Verify installation
uv run python main.py --help
```

### Step 3: Verify Setup

Ensure everything is working correctly:

```bash
# Check UV installation
uv --version

# Check Python version
uv run python --version

# Test the script help
uv run python main.py --help

# List available models in LM Studio (optional)
# Check your LM Studio interface for loaded models
```

## Quick Start

### Process Your First Images (CLI)

1. **Create an input directory** and add your images:
   ```bash
   mkdir input
   # Copy your images to the input directory
   ```

2. **Start LM Studio** and load your preferred vision-language model

3. **Run the OCR tool**:
   ```bash
   # Process all formats for all images
   uv run python main.py --format all --input-dir input
   ```

4. **Check the output**:
   ```bash
   ls -la output/
   ```

### Run the Web UI (Gradio)

1. Ensure dependencies are installed:
   ```bash
   uv sync
   ```
2. Start LM Studio and load your preferred vision-language model
3. Launch the UI:
   ```bash
   uv run python ui.py
   ```
4. In your browser, open the printed local URL (e.g., http://127.0.0.1:7860)
5. Choose input source (folder or upload), select format, set the model name, then click "Run OCR"

Notes:
- The UI accepts any LM Studio vision model name (e.g., `chandra-ocr`, `qwen2.5-vl-7b`, `gemma-3-12b`).
- Outputs appear under `output/` as with the CLI and are available in the UI for download and preview.

## Usage

### Basic Usage

```bash
# Process all formats for all images
uv run python main.py --format all --input-dir input_images

# Process specific format only
uv run python main.py --format html --input-dir input_images
uv run python main.py --format markdown_labels --input-dir input_images
uv run python main.py --format markdown_no_labels --input-dir input_images
uv run python main.py --format images --input-dir input_images
```

### CLI Options

| Option | Description | Default |
|---------|-------------|----------|
| `--format` | Output format to generate (markdown_labels, markdown_no_labels, html, images, all) | `all` |
| `--input-dir` | Input directory containing images | `input` |
| `--model` | LM Studio model name (any vision-language model) | `chandra-ocr` |

### Output Formats

| Format | Description | Output File |
|--------|-------------|--------------|
| `markdown_labels` | Markdown with page headers and footers | `{filename}.md` |
| `markdown_no_labels` | Markdown without page headers/footers | `{filename}.md` |
| `html` | Styled HTML with proper structure | `{filename}.html` |
| `images` | Images with bounding boxes | `{filename}_bboxes.png` |
| `all` | Generate all formats | All above files |

### Output Structure

```
output/
├── logs/                           # Processing logs with timestamps
│   └── ocr_processing_YYYYMMDD_HHMMSS.log
├── markdown_labels/                # Markdown output with headers/footers
│   └── image1.md
├── markdown_no_labels/             # Markdown output without headers/footers  
│   └── image1.md
├── html/                           # HTML output with styling
│   └── image1.html
└── images/                         # Images with bounding boxes
    └── image1_bboxes.png
```

### Merging Markdown Files

After processing, you can merge all markdown files into a single document:

```bash
# Merge markdown with labels
uv run python merge_markdown.py output/markdown_labels merged_with_labels.md

# Merge markdown without labels  
uv run python merge_markdown.py output/markdown_no_labels merged_without_labels.md

# Merge all markdown files (sorted by filename)
uv run python merge_markdown.py output/markdown_no_labels all_merged.md
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
# Use Qwen2.5-VL for multilingual support
uv run python main.py --format all --model qwen2.5-vl-7b

# Use Gemma-3 for general OCR tasks
uv run python main.py --format all --model gemma-3-7b

# Use any custom loaded model
uv run python main.py --format all --model your-model-name
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

## Troubleshooting

### Installation Issues

#### UV Installation Problems

**macOS/Linux:**
```bash
# If curl command fails, try:
curl -LsSf https://astral.sh/uv/install.sh | sh

# If permissions error, try:
sudo curl -LsSf https://astral.sh/uv/install.sh | sh

# Alternative via pip:
pip install uv
```

**Windows:**
```powershell
# If PowerShell script is blocked, try:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Alternative via pip:
pip install uv
```

#### Git Cloning Problems

**Permission Denied:**
```bash
# Check if you have the correct repository URL
git remote -v

# If using SSH, ensure your SSH key is set up
ssh -T git@github.com
```

**Network Issues:**
```bash
# Try cloning with HTTPS instead of SSH
git clone https://github.com/YOUR-USERNAME/lmstudio_ocr.git
```

### Runtime Issues

#### LM Studio Connection
1. **Ensure LM Studio is running** locally
2. **Check that your vision-language model is loaded** in LM Studio
3. **Verify the model name matches** the `--model` parameter exactly
4. **Check LM Studio's API endpoint** (default: http://localhost:1234)
5. **Ensure the model supports vision input** (not all models do)

#### No Images Found
```bash
# Verify input directory exists
ls -la input/

# Check image file extensions (PNG, JPG, JPEG, TIFF)
find input/ -type f -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.tiff" -o -name "*.tif"
```

#### Permission Errors
```bash
# Ensure write access to output directory
chmod 755 output/

# Check file permissions for input images
ls -la input/
```

#### Processing Timeouts
- Large images may take longer to process
- Check LM Studio resource usage
- Consider reducing image size or processing in batches

### Log Analysis
Check detailed logs in `output/logs/` for:
- Processing errors
- Performance metrics
- File-specific issues

```bash
# View the latest log file
ls -la output/logs/
tail -f output/logs/ocr_processing_*.log
```

## Advanced Configuration

### Custom Model Settings
You can use any vision-language model available in LM Studio by modifying the `--model` parameter:

```bash
# Use specialized OCR models
uv run python main.py --format all --model chandra-ocr
uv run python main.py --format all --model nougat

# Use general vision models
uv run python main.py --format all --model qwen2.5-vl-7b
uv run python main.py --format all --model gemma-3-12b
uv run python main.py --format all --model internvl-1.5

# Use any custom loaded model
uv run python main.py --format all --model your-model-name
```

#### Model Performance Tips
- **Smaller models** (3B-7B): Faster processing, good for simple documents
- **Medium models** (12B-30B): Balance of speed and accuracy
- **Larger models** (70B+): Best accuracy but slower, requires more RAM
- **Specialized OCR models**: Usually better for text-heavy documents
- **General vision models**: Better for mixed content (text + images + diagrams)

#### Finding Model Names
1. Open LM Studio
2. Go to the "Models" tab
3. Look at your downloaded models
4. Use the exact name as shown in LM Studio
5. The model name is case-sensitive

### Batch Processing Tips
- **Organize images** in subdirectories for better organization
- **Use descriptive filenames** for easier identification
- **Monitor system resources** when processing large batches
- **Consider processing in smaller batches** for very large image collections

## Requirements

- Python 3.13+
- LM Studio running with a compatible vision-language model
- UV package manager
- Git (for cloning the repository)

## Output Quality

The tool uses optimized OCR prompts that work with any vision-language model:
- ✅ High accuracy text recognition
- ✅ Proper layout preservation
- ✅ Structured HTML output
- ✅ Bounding box information (when supported by model)
- ✅ Multi-language support (model dependent)
- ✅ Table and form handling
- ✅ Compatible with various model architectures

### Model-Specific Considerations
- **Specialized OCR models**: Best for text-heavy documents and complex layouts
- **General vision models**: Good for mixed content but may vary in OCR accuracy
- **Multilingual models**: Better for non-English text processing
- **Model size affects**: Processing speed, accuracy, and memory usage

## License

This tool is designed to work with any vision-language model in LM Studio. When using specialized OCR models like Chandra OCR, please respect their respective licenses and guidelines:

- **Chandra OCR**: Follows Datalab's OCR guidelines (https://github.com/datalab-to/chandra)
- **Other Models**: Respect each model's specific license terms
- **This Tool**: Open source - see repository license for details

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the logs in `output/logs/`
3. Open an issue on GitHub with detailed information about your problem