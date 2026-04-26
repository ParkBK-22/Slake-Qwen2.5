from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from PIL import Image

from src.utils.image_utils import to_pil_image


@dataclass
class SlakeSample:
    image: Image.Image | None
    question: str
    answer: str | list[str] | None
    sample_id: str
    image_id: str | None
    split: str
    raw_keys: list[str]


IMAGE_KEYS = ["image", "img", "picture", "pil_image"]
QUESTION_KEYS = ["question", "q", "Question", "query"]
ANSWER_KEYS = ["answer", "answers", "gt_answer", "label", "Answer"]
SAMPLE_ID_KEYS = ["sample_id", "id", "qid", "question_id", "qa_id"]
IMAGE_ID_KEYS = ["image_id", "img_id", "image_name", "img_name", "filename", "file_name"]


def _first_existing(sample: dict[str, Any], keys: list[str]) -> tuple[str | None, Any]:
    for key in keys:
        if key in sample and sample[key] is not None:
            return key, sample[key]
    return None, None


def _stringify(value: Any) -> str | list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(v) for v in value]
    return str(value)


def load_slake_dataset(dataset_repo: str, split: str | None = None) -> Dataset | DatasetDict:
    """Load a SLAKE-like Hugging Face dataset.

    The default target is mdwiratathya/SLAKE-vqa-english, but this function keeps
    the repo name configurable because public SLAKE mirrors use different schemas.
    """
    if split is None:
        return load_dataset(dataset_repo)
    return load_dataset(dataset_repo, split=split)


def get_split(ds: Dataset | DatasetDict, split: str) -> Dataset:
    """Return requested split, with a practical fallback for small HF mirrors."""
    if isinstance(ds, Dataset):
        return ds

    if split in ds:
        return ds[split]

    aliases = {
        "validation": ["val", "valid", "dev"],
        "val": ["validation", "valid", "dev"],
        "test": ["test", "validation", "val", "train"],
        "train": ["train"],
    }
    for candidate in aliases.get(split, []):
        if candidate in ds:
            print(f"[WARN] split='{split}' not found. Using split='{candidate}' instead.")
            return ds[candidate]

    available = list(ds.keys())
    raise KeyError(f"Split '{split}' not found. Available splits: {available}")


def extract_slake_sample(sample: dict[str, Any], idx: int | None = None, split: str = "unknown") -> SlakeSample:
    """Extract image, question, answer, sample_id, and image_id from a SLAKE sample.

    Different SLAKE Hugging Face mirrors use different column names. This function
    searches common key names and falls back conservatively.
    """
    raw_keys = list(sample.keys())

    image_key, image_obj = _first_existing(sample, IMAGE_KEYS)
    question_key, question = _first_existing(sample, QUESTION_KEYS)
    answer_key, answer = _first_existing(sample, ANSWER_KEYS)
    sample_id_key, sample_id = _first_existing(sample, SAMPLE_ID_KEYS)
    image_id_key, image_id = _first_existing(sample, IMAGE_ID_KEYS)

    # Fallback: some datasets use conversations/messages. Keep this minimal because
    # the experiment requires the original dataset question without extra instruction.
    if question is None:
        for key in raw_keys:
            lowered = key.lower()
            if "question" in lowered and sample[key] is not None:
                question_key, question = key, sample[key]
                break

    if answer is None:
        for key in raw_keys:
            lowered = key.lower()
            if "answer" in lowered and sample[key] is not None:
                answer_key, answer = key, sample[key]
                break

    if image_obj is None:
        for key in raw_keys:
            lowered = key.lower()
            if ("image" in lowered or "img" in lowered) and sample[key] is not None:
                image_key, image_obj = key, sample[key]
                break

    if question is None:
        raise ValueError(f"Could not find question column. Raw keys: {raw_keys}")

    if sample_id is None:
        sample_id = idx if idx is not None else "unknown"

    pil_image = None
    if image_obj is not None:
        pil_image = to_pil_image(image_obj)

    return SlakeSample(
        image=pil_image,
        question=str(question),
        answer=_stringify(answer),
        sample_id=str(sample_id),
        image_id=None if image_id is None else str(image_id),
        split=split,
        raw_keys=raw_keys,
    )
