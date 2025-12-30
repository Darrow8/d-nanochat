#!/usr/bin/env python3
"""Merge all Shakespeare .txt files into a single output file."""

from pathlib import Path


def merge_shakespeare_texts(
    input_dir: Path | str = None,
    output_file: Path | str = None,
    separator: str = "\n\n" + "=" * 80 + "\n\n",
) -> str:
    """
    Merge all .txt files from the shakespeare directory.
    
    Args:
        input_dir: Directory containing .txt files. Defaults to ../shakespeare/
        output_file: Output file path. If None, only returns merged content.
        separator: String to insert between each file's content.
    
    Returns:
        The merged text content.
    """
    if input_dir is None:
        # Default: shakespeare folder is sibling to d-nanochat in nanochat-project
        input_dir = Path(__file__).parent.parent.parent / "shakespeare"
    else:
        input_dir = Path(input_dir)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Get all .txt files, sorted alphabetically
    txt_files = sorted(input_dir.glob("*.txt"))
    
    if not txt_files:
        raise ValueError(f"No .txt files found in {input_dir}")
    
    print(f"Found {len(txt_files)} text files in {input_dir}")
    
    merged_parts = []
    for txt_file in txt_files:
        print(f"  Reading: {txt_file.name}")
        content = txt_file.read_text(encoding="utf-8")
        merged_parts.append(content)
    
    merged_content = separator.join(merged_parts)
    
    if output_file is not None:
        output_file = Path(output_file)
        output_file.write_text(merged_content, encoding="utf-8")
        print(f"\nMerged content written to: {output_file}")
        print(f"Total size: {len(merged_content):,} characters")
    
    return merged_content


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Merge Shakespeare text files")
    parser.add_argument(
        "-i", "--input-dir",
        type=Path,
        default=None,
        help="Input directory containing .txt files (default: ../shakespeare/)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path(__file__).parent / "shakespeare_merged.txt",
        help="Output file path (default: shakespeare_merged.txt in same directory)",
    )
    parser.add_argument(
        "--no-separator",
        action="store_true",
        help="Don't add separators between files",
    )
    
    args = parser.parse_args()
    
    separator = "\n\n" if args.no_separator else "\n\n" + "=" * 80 + "\n\n"
    
    merge_shakespeare_texts(
        input_dir=args.input_dir,
        output_file=args.output,
        separator=separator,
    )

