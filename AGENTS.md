# LM Studio OCR - Agent Guidelines

## Commands
- Install dependencies: `uv sync`
- Run main script: `uv run python main.py --format all --input-dir input`
- Lint: `uv run ruff check .`
- Format: `uv run ruff format .`
- Type check: `uv run mypy .`
- Run tests: `uv run pytest`
- Run single test: `uv run pytest path/to/test.py::test_name`

## Script Usage
- `--format`: Choose output format (markdown_with_headers, markdown, html, images, all)
- `--input-dir`: Specify input directory (default: input)
- `--model`: Specify LM Studio model (default: chandra-ocr)
- Script processes images one-by-one and skips already processed files
- Progress is tracked by checking existing output files in folders
- Comprehensive logging is saved to `output/logs/`
- Merged markdown files are saved to `output/merged_markdown/`

## Code Style
- Python 3.13+ required
- Use `uv` for dependency management
- Follow PEP 8 formatting (enforced by ruff)
- Use type hints for all function signatures
- Import standard library first, then third-party packages, then local modules
- Use descriptive variable names in snake_case
- Handle exceptions with specific error types
- Keep functions focused and under 20 lines when possible
- Use f-strings for string formatting
- Add docstrings for public functions and classes
- Use dataclasses for structured data
- Process images with PIL/Pillow
- Parse HTML with BeautifulSoup
- Use proper Chandra OCR prompts from `prompts.py`
- Implement folder-based progress checking (no separate tracking files)
- Add comprehensive logging with timestamps