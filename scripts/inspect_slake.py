from __future__ import annotations

import argparse
import pprint
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.datasets.load_slake import extract_slake_sample, get_split, load_slake_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect SLAKE Hugging Face dataset schema.")
    parser.add_argument("--dataset_repo", type=str, default="mdwiratathya/SLAKE-vqa-english")
    parser.add_argument("--split", type=str, default="test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ds_all = load_slake_dataset(args.dataset_repo, split=None)

    print("=" * 80)
    print(f"Dataset repo: {args.dataset_repo}")
    print("=" * 80)

    if hasattr(ds_all, "keys"):
        print(f"Split names: {list(ds_all.keys())}")
    else:
        print("Split names: dataset object has no split dictionary")

    ds = get_split(ds_all, args.split)
    print(f"Selected split: {args.split}")
    print(f"Number of samples: {len(ds)}")
    print(f"Dataset columns: {ds.column_names}")

    first = ds[0]
    print("\nFirst raw sample:")
    pprint.pp(first, depth=2, width=120)

    print("\nExtracted first sample:")
    extracted = extract_slake_sample(first, idx=0, split=args.split)
    printable = {
        "image": None if extracted.image is None else f"PIL.Image size={extracted.image.size}, mode={extracted.image.mode}",
        "question": extracted.question,
        "answer": extracted.answer,
        "sample_id": extracted.sample_id,
        "image_id": extracted.image_id,
        "split": extracted.split,
        "raw_keys": extracted.raw_keys,
    }
    pprint.pp(printable, depth=3, width=120)

    print("\nDetected locations:")
    print("- image: extracted.image")
    print("- question: extracted.question")
    print("- answer: extracted.answer")
    print("- sample_id: extracted.sample_id")
    print("- image_id: extracted.image_id")


if __name__ == "__main__":
    main()
