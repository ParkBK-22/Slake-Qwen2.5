from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any

from datasets import DatasetDict
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.datasets.load_slake import extract_slake_sample, get_split, load_slake_dataset
from src.models.qwen_vl import Qwen25VLWrapper, QwenVLConfig
from src.utils.image_utils import (
    make_black_image,
    make_patchshuffle_image,
    resize_image,
)
from src.utils.io_utils import append_jsonl, ensure_dir, get_completed_keys
from src.utils.seed_utils import set_seed

VALID_CONDITIONS = ["original", "black", "no_image", "patchshuffle_16"]

PROMPT_INSTRUCTION = (
    "If you cannot answer from the image, say 'unknown' or describe what is wrong with the image."
)


def build_prompt(question: str) -> str:
    """
    Build the exact user text sent to Qwen.

    Current experiment prompt:
        {question}
        If you cannot answer from the image, say 'unknown' or describe what is wrong with the image.
    """
    question = str(question).strip()
    return f"{question}\n{PROMPT_INSTRUCTION}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen2.5-VL inference on SLAKE.")

    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
    )
    parser.add_argument(
        "--dataset_repo",
        type=str,
        default="mdwiratathya/SLAKE-vqa-english",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split. Use 'test' for the full test split.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=VALID_CONDITIONS,
        choices=VALID_CONDITIONS,
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to run. Default: all samples in the selected split.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=16,
        help="Patch size for patchshuffle condition.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/raw",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only inspect data/model inputs. Do not load model.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed (condition, unique_sample_key) rows.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file before running. Recommended for a fresh full run.",
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Save resized condition images for debugging.",
    )

    return parser.parse_args()


def make_output_path(
    output_dir: str | Path,
    dataset_repo: str,
    split: str,
    max_new_tokens: int,
    image_size: int,
    patch_size: int,
) -> Path:
    """
    Return a clean output filename for the current experiment.

    Current experiment:
    - Dataset: SLAKE
    - Split: test
    - Conditions: original / black / no_image / patchshuffle_16
    - Prompt: unknown-or-describe
    - max_new_tokens: 128
    """
    return Path(output_dir) / "slake_qwen_unknown_prompt.jsonl"


def normalize_for_key(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def make_unique_sample_key(sample: Any) -> str:
    """
    Unique key for avoiding duplicated output rows.

    This identifies the dataset sample, not the question-answer content.
    Therefore, repeated question-answer pairs are preserved.
    """
    split = normalize_for_key(sample.split)
    sample_id = normalize_for_key(sample.sample_id)
    image_id = normalize_for_key(sample.image_id)

    if image_id:
        return f"{split}::{sample_id}::{image_id}"

    return f"{split}::{sample_id}"


def collect_samples(ds_all: Any, split: str) -> list[tuple[str, int, Any]]:
    """
    Return list of (split_name, idx, raw_sample).

    For this experiment, use:
        --split test
    """
    items: list[tuple[str, int, Any]] = []

    if split == "all":
        if isinstance(ds_all, DatasetDict):
            split_names = list(ds_all.keys())
            print(f"[INFO] Running all splits: {split_names}")

            for split_name in split_names:
                ds = ds_all[split_name]
                for idx in range(len(ds)):
                    items.append((split_name, idx, ds[idx]))
        else:
            print("[WARN] Dataset is not DatasetDict. Treating it as a single split.")
            for idx in range(len(ds_all)):
                items.append(("all", idx, ds_all[idx]))
    else:
        ds = get_split(ds_all, split)
        for idx in range(len(ds)):
            items.append((split, idx, ds[idx]))

    return items


def build_row_base(
    *,
    args: argparse.Namespace,
    condition: str,
    sample: Any,
    unique_sample_key: str,
    input_prompt: str,
) -> dict[str, Any]:
    return {
        "model_name": args.model_name,
        "dataset": args.dataset_repo,
        "condition": condition,
        "unique_sample_key": unique_sample_key,
        "sample_id": sample.sample_id,
        "question": sample.question,
        "input_prompt": input_prompt,
        "prompt_instruction": PROMPT_INSTRUCTION,
        "gt_answer": sample.answer,
        "pred_answer": None,
        "image_size": args.image_size,
        "patch_size": args.patch_size if condition == "patchshuffle_16" else None,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "temperature": 0.0,
        "random_seed": args.random_seed,
        "split": sample.split,
        "image_id": sample.image_id,
        "raw_sample_keys": sample.raw_keys,
        "error": None,
    }


def save_debug_image(image: Any, condition: str, sample_key: str) -> str:
    image_dir = ensure_dir(REPO_ROOT / "outputs" / "images" / condition)
    safe_sample_key = str(sample_key).replace("/", "_").replace(":", "_")
    path = image_dir / f"{safe_sample_key}.png"
    image.save(path)
    return str(path)


def main() -> None:
    args = parse_args()
    set_seed(args.random_seed)

    output_path = make_output_path(
        output_dir=args.output_dir,
        dataset_repo=args.dataset_repo,
        split=args.split,
        max_new_tokens=args.max_new_tokens,
        image_size=args.image_size,
        patch_size=args.patch_size,
    )
    ensure_dir(output_path.parent)

    if args.overwrite and output_path.exists():
        print(f"[INFO] Removing existing output file: {output_path}")
        output_path.unlink()

    completed = get_completed_keys(output_path) if args.resume else set()

    print(f"[INFO] Loading dataset: {args.dataset_repo}")
    ds_all = load_slake_dataset(args.dataset_repo, split=None)

    raw_items = collect_samples(ds_all, args.split)
    print(f"[INFO] Raw samples found: {len(raw_items)}")

    samples = []
    seen_sample_keys = set()

    for split_name, idx, raw_sample in raw_items:
        try:
            sample = extract_slake_sample(raw_sample, idx=idx, split=split_name)
            unique_sample_key = make_unique_sample_key(sample)

            if unique_sample_key in seen_sample_keys:
                continue

            seen_sample_keys.add(unique_sample_key)
            samples.append(sample)

        except Exception as exc:
            error_row = {
                "model_name": args.model_name,
                "dataset": args.dataset_repo,
                "condition": "extract_error",
                "unique_sample_key": f"{split_name}::{idx}",
                "sample_id": str(idx),
                "question": None,
                "input_prompt": None,
                "prompt_instruction": PROMPT_INSTRUCTION,
                "gt_answer": None,
                "pred_answer": None,
                "image_size": args.image_size,
                "patch_size": None,
                "max_new_tokens": args.max_new_tokens,
                "do_sample": False,
                "temperature": 0.0,
                "random_seed": args.random_seed,
                "split": split_name,
                "image_id": None,
                "raw_sample_keys": list(raw_sample.keys())
                if isinstance(raw_sample, dict)
                else None,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }

            if not args.dry_run:
                append_jsonl(output_path, error_row)

    if args.num_samples is not None:
        samples = samples[: args.num_samples]

    print(f"[INFO] Unique dataset samples to run: {len(samples)}")
    print(f"[INFO] Conditions: {args.conditions}")
    print(f"[INFO] Prompt instruction: {PROMPT_INSTRUCTION}")
    print(f"[INFO] Max new tokens: {args.max_new_tokens}")
    print(f"[INFO] Output: {output_path}")

    model = None
    if not args.dry_run:
        print(f"[INFO] Loading model: {args.model_name}")
        model = Qwen25VLWrapper(QwenVLConfig(model_name=args.model_name))
    else:
        print("[DRY RUN] Model will not be loaded. Rows will not be written.")

    for sample in tqdm(samples, desc="SLAKE inference"):
        unique_sample_key = make_unique_sample_key(sample)
        input_prompt = build_prompt(sample.question)

        if args.dry_run:
            print("\n[DRY RUN SAMPLE]")
            print(f"unique_sample_key={unique_sample_key}")
            print(f"split={sample.split}")
            print(f"sample_id={sample.sample_id}")
            print(f"image_id={sample.image_id}")
            print(f"question={sample.question}")
            print(f"input_prompt={input_prompt}")
            print(f"answer={sample.answer}")
            print(f"has_image={sample.image is not None}")
            continue

        for condition in args.conditions:
            completed_key = (condition, unique_sample_key)

            if args.resume and completed_key in completed:
                continue

            row = build_row_base(
                args=args,
                condition=condition,
                sample=sample,
                unique_sample_key=unique_sample_key,
                input_prompt=input_prompt,
            )

            try:
                if condition == "no_image":
                    input_image = None

                else:
                    if sample.image is None:
                        raise ValueError(
                            "This condition requires an image, but image is missing."
                        )

                    resized = resize_image(sample.image, size=args.image_size)

                    if condition == "original":
                        input_image = resized

                    elif condition == "black":
                        input_image = make_black_image(resized)

                    elif condition == "patchshuffle_16":
                        input_image = make_patchshuffle_image(
                            resized,
                            patch_size=args.patch_size,
                            seed=args.random_seed,
                        )

                    else:
                        raise ValueError(f"Unknown condition: {condition}")

                    if args.save_images:
                        row["saved_image_path"] = save_debug_image(
                            image=input_image,
                            condition=condition,
                            sample_key=unique_sample_key,
                        )

                assert model is not None

                pred = model.generate(
                    question=input_prompt,
                    image=input_image,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                )

                row["pred_answer"] = pred

            except Exception as exc:
                row["error"] = repr(exc)
                row["traceback"] = traceback.format_exc()

            append_jsonl(output_path, row)

    print(f"[DONE] Saved results to: {output_path}")


if __name__ == "__main__":
    main()