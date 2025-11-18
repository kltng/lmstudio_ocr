#!/usr/bin/env python3
"""
Markdown Merger Script
Merges all markdown files from a directory into a single comprehensive markdown file.
Saves output to output/merged_markdown/ subfolder.
"""

import argparse
import sys
from pathlib import Path


def merge_markdown_files(input_dir: str, output_filename: str, sort_by_filename: bool = True) -> None:
    """Merge all markdown files from input directory into a single file"""
    input_path = Path(input_dir)

    if not input_path.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        sys.exit(1)

    # Find all markdown files
    md_files = list(input_path.glob("*.md"))

    if not md_files:
        print(f"No markdown files found in {input_dir}")
        return

    # Sort files by filename (natural sort)
    if sort_by_filename:
        md_files.sort(key=lambda x: x.name)

    print(f"Found {len(md_files)} markdown files to merge")

    # Create merged content
    merged_content = []
    merged_content.append("# Merged OCR Results\n")
    merged_content.append(f"*Generated from {len(md_files)} files*\n")
    merged_content.append("---\n\n")

    for i, md_file in enumerate(md_files, 1):
        print(f"Processing {i}/{len(md_files)}: {md_file.name}")

        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if content:
                # Add file header
                filename = md_file.stem
                merged_content.append(f"## {filename}\n\n")
                merged_content.append(content)
                merged_content.append("\n---\n\n")

        except (IOError, OSError, PermissionError) as e:
            print(f"Error reading file {md_file}: {e}")
        except UnicodeDecodeError as e:
            print(f"Error decoding file {md_file} (not UTF-8?): {e}")
        except Exception as e:
            print(f"Unexpected error reading {md_file}: {e}")

    # Create output directory and write merged file
    try:
        # Create merged_markdown directory in output folder
        output_dir = Path("output/merged_markdown")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / output_filename

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("".join(merged_content))

        print(f"Successfully merged {len(md_files)} files into {output_file}")

    except (IOError, OSError, PermissionError) as e:
        print(f"Error writing merged file to {output_file}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error writing merged file: {e}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge markdown files into a single file"
    )
    parser.add_argument(
        "input_dir", help="Directory containing markdown files to merge"
    )
    parser.add_argument(
        "output_filename",
        nargs="?",
        default="merged_markdown.md",
        help="Output filename for merged markdown (default: merged_markdown.md)",
    )
    parser.add_argument(
        "--no-sort", action="store_true", help="Do not sort files by filename"
    )

    args = parser.parse_args()

    sort_files = not args.no_sort
    merge_markdown_files(args.input_dir, args.output_filename, sort_files)


if __name__ == "__main__":
    main()
