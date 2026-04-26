from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.datasets.load_slake import extract_slake_sample, get_split, load_slake_dataset
from src.models.qwen_vl import Qwen25VLWrapper, QwenVLConfig
from src.utils.image_utils import make_black_image, resize_image
from src.utils.io_utils import append_jsonl, ensure_dir, get_completed_keys
from src.utils.seed_utils import set_seed

VALID_CONDITIONS = ["original", "black", "text_only"]
VALID_PROMPT_STYLES = ["raw", "one_word"]


def build_prompt(question: str, prompt_style: str) -> str:
    """
    Build the exact user text sent to Qwen.

    raw:
        Use the dataset question as-is.

    one_word:
        Add a minimal answer-format instruction.
    """
    question = str(question).strip()

    if prompt_style == "raw":
        return question

    if prompt_style == "one_word":
        return f"{question}\nAnswer in one word."

    raise ValueError(f"Unknown prompt_style: {prompt_style}")


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
        help="Number of samples to run. Default: all samples.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--prompt_style",
        type=str,
        default="one_word",
        choices=VALID_PROMPT_STYLES,
        help=(
            "Prompt style. "
            "raw: use the dataset question only. "
            "one_word: append 'Answer in one word.'"
        ),
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
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
        help="Skip completed (condition, sample_id) rows.",
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
    prompt_style: str,
) -> Path:
    safe_dataset = dataset_repo.replace("/", "_").replace("-", "_")
    return Path(output_dir) / (
        f"qwen_slake_{safe_dataset}_{split}_{prompt_style}_mnt{max_new_tokens}.jsonl"
    )


def build_row_base(
    *,
    args: argparse.Namespace,
    condition: str,
    sample: Any,
    input_prompt: str,
) -> dict[str, Any]:
    return {
        "model_name": args.model_name,
        "dataset": args.dataset_repo,
        "condition": condition,
        "sample_id": sample.sample_id,
        "question": sample.question,
        "input_prompt": input_prompt,
        "prompt_style": args.prompt_style,
        "gt_answer": sample.answer,
        "pred_answer": None,
        "image_size": args.image_size,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "temperature": 0.0,
        "random_seed": args.random_seed,
        "split": sample.split,
        "image_id": sample.image_id,
        "raw_sample_keys": sample.raw_keys,
        "error": None,
    }


def save_debug_image(image: Any, condition: str, sample_id: str) -> str:
    image_dir = ensure_dir(REPO_ROOT / "outputs" / "images" / condition)
    path = image_dir / f"{sample_id}.png"
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
        prompt_style=args.prompt_style,
    )
    ensure_dir(output_path.parent)

    completed = get_completed_keys(output_path) if args.resume else set()

    print(f"[INFO] Loading dataset: {args.dataset_repo}")
    ds_all = load_slake_dataset(args.dataset_repo, split=None)
    ds = get_split(ds_all, args.split)

    total = len(ds) if args.num_samples is None else min(args.num_samples, len(ds))

    print(f"[INFO] Split: {args.split}")
    print(f"[INFO] Total samples to visit: {total}")
    print(f"[INFO] Conditions: {args.conditions}")
    print(f"[INFO] Prompt style: {args.prompt_style}")
    print(f"[INFO] Output: {output_path}")

    model = None
    if not args.dry_run:
        print(f"[INFO] Loading model: {args.model_name}")
        model = Qwen25VLWrapper(QwenVLConfig(model_name=args.model_name))
    else:
        print("[DRY RUN] Model will not be loaded. Rows will not be written.")

    for idx in tqdm(range(total), desc="SLAKE inference"):
        raw_sample = ds[idx]

        try:
            sample = extract_slake_sample(raw_sample, idx=idx, split=args.split)
        except Exception as exc:
            error_row = {
                "model_name": args.model_name,
                "dataset": args.dataset_repo,
                "condition": "extract_error",
                "sample_id": str(idx),
                "question": None,
                "input_prompt": None,
                "prompt_style": args.prompt_style,
                "gt_answer": None,
                "pred_answer": None,
                "image_size": args.image_size,
                "max_new_tokens": args.max_new_tokens,
                "do_sample": False,
                "temperature": 0.0,
                "random_seed": args.random_seed,
                "split": args.split,
                "image_id": None,
                "raw_sample_keys": list(raw_sample.keys())
                if isinstance(raw_sample, dict)
                else None,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            append_jsonl(output_path, error_row)
            continue

        input_prompt = build_prompt(sample.question, args.prompt_style)

        if args.dry_run and idx < 3:
            print("\n[DRY RUN SAMPLE]")
            print(f"idx={idx}")
            print(f"sample_id={sample.sample_id}")
            print(f"image_id={sample.image_id}")
            print(f"question={sample.question}")
            print(f"input_prompt={input_prompt}")
            print(f"answer={sample.answer}")
            print(f"has_image={sample.image is not None}")
            continue

        for condition in args.conditions:
            if args.resume and (condition, sample.sample_id) in completed:
                continue

            row = build_row_base(
                args=args,
                condition=condition,
                sample=sample,
                input_prompt=input_prompt,
            )

            try:
                if condition == "text_only":
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
                    else:
                        raise ValueError(f"Unknown condition: {condition}")

                    if args.save_images:
                        row["saved_image_path"] = save_debug_image(
                            image=input_image,
                            condition=condition,
                            sample_id=sample.sample_id,
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