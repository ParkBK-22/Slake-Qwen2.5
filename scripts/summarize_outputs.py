from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.io_utils import ensure_dir, save_text

UNKNOWN_KEYWORDS = [
    "unknown",
    "cannot determine",
    "can't determine",
    "not enough information",
    "unable to tell",
    "cannot tell",
    "not visible",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize raw Qwen SLAKE JSONL outputs without scoring.")
    parser.add_argument("--input", type=str, default=None, help="One JSONL file. If omitted, reads outputs/raw/*.jsonl")
    parser.add_argument("--input_dir", type=str, default="outputs/raw")
    parser.add_argument("--output_dir", type=str, default="outputs/tables")
    parser.add_argument("--top_k", type=int, default=20)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_unknown_like(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in UNKNOWN_KEYWORDS)


def main() -> None:
    args = parse_args()
    if args.input:
        paths = [Path(args.input)]
    else:
        paths = sorted(Path(args.input_dir).glob("*.jsonl"))

    if not paths:
        raise FileNotFoundError("No JSONL files found. Use --input or check --input_dir.")

    rows = []
    for path in paths:
        file_rows = read_jsonl(path)
        for row in file_rows:
            row["source_file"] = str(path)
        rows.extend(file_rows)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No rows loaded from JSONL files.")

    # Keep failed rows visible, but summarize normal condition rows separately.
    if "pred_answer" not in df.columns:
        df["pred_answer"] = ""
    df["pred_answer"] = df["pred_answer"].fillna("").astype(str)
    df["pred_len_chars"] = df["pred_answer"].str.len()
    df["pred_len_words"] = df["pred_answer"].apply(lambda x: len(x.split()))
    df["unknown_like"] = df["pred_answer"].apply(is_unknown_like)
    df["has_error"] = df.get("error", pd.Series([None] * len(df))).notna() & (df.get("error", "") != "")

    group_cols = ["condition"]
    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            num_rows=("pred_answer", "size"),
            num_errors=("has_error", "sum"),
            avg_pred_len_chars=("pred_len_chars", "mean"),
            avg_pred_len_words=("pred_len_words", "mean"),
            unknown_like_count=("unknown_like", "sum"),
            unknown_like_ratio=("unknown_like", "mean"),
        )
        .reset_index()
    )

    output_dir = ensure_dir(args.output_dir)
    csv_path = output_dir / "output_summary.csv"
    md_path = output_dir / "output_summary.md"
    summary.to_csv(csv_path, index=False)

    md_lines = ["# Output Summary", "", summary.to_markdown(index=False), ""]
    md_lines.append("## Frequent Answers by Condition")
    md_lines.append("")
    for condition, sub in df.groupby("condition", dropna=False):
        md_lines.append(f"### {condition}")
        answers = [a.strip() for a in sub["pred_answer"].tolist() if a.strip()]
        counts = Counter(answers).most_common(args.top_k)
        if counts:
            top_df = pd.DataFrame(counts, columns=["pred_answer", "count"])
            md_lines.append(top_df.to_markdown(index=False))
        else:
            md_lines.append("No non-empty predictions.")
        md_lines.append("")

    save_text(md_path, "\n".join(md_lines))
    print(f"[DONE] Saved CSV summary to: {csv_path}")
    print(f"[DONE] Saved Markdown summary to: {md_path}")


if __name__ == "__main__":
    main()
