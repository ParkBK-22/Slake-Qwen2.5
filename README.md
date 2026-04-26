# Qwen2.5-VL SLAKE Hallucination Inference

This repository runs a simple VQA hallucination check on the English subset of SLAKE using `Qwen/Qwen2.5-VL-7B-Instruct`.

The current experiment saves model outputs only. It does **not** score correctness.

## Experiment conditions

For each SLAKE sample, the script can run three conditions:

1. `original`: original image + question
2. `black`: black image + question
3. `text_only`: question only

The prompt is exactly the dataset question. No extra instruction is appended.

## Repository structure

```text
qwen-slake-hallucination/
  README.md
  requirements.txt
  .gitignore
  configs/
    experiment.yaml
  scripts/
    inspect_slake.py
    run_inference.py
    summarize_outputs.py
  src/
    datasets/
      load_slake.py
    models/
      qwen_vl.py
    utils/
      image_utils.py
      io_utils.py
      seed_utils.py
  outputs/
    raw/
    tables/
    images/
```

## 1. Vast.ai server preparation

Choose a GPU instance that can run a 7B vision-language model. A 24GB GPU may work depending on the environment and attention implementation, but a larger GPU such as A100 40GB/80GB is safer.

Recommended base environment:

```bash
nvidia-smi
python --version
```

Use Python 3.10 or newer.

## 2. Create a Git repository

On the Vast.ai server:

```bash
mkdir -p ~/projects
cd ~/projects
git clone <YOUR_GITHUB_REPO_URL> qwen-slake-hallucination
cd qwen-slake-hallucination
```

If you are starting from these files directly:

```bash
cd ~/projects/qwen-slake-hallucination
git init
git add .
git commit -m "Initial Qwen SLAKE hallucination experiment"
```

## 3. Connect with VSCode Remote SSH

On your local machine:

1. Install the VSCode extension: `Remote - SSH`.
2. Add the Vast.ai SSH command to your SSH config.
3. Connect to the server from VSCode.
4. Open the repository folder, for example:

```text
~/projects/qwen-slake-hallucination
```

## 4. Python environment setup

On the server:

```bash
cd ~/projects/qwen-slake-hallucination
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 5. Hugging Face login

Some models or datasets may require Hugging Face access.

```bash
huggingface-cli login
```

Then paste your Hugging Face token.

## 6. Install requirements

```bash
pip install -r requirements.txt
```

If you encounter CUDA/PyTorch mismatch issues, install the PyTorch build that matches your CUDA version first, then reinstall the remaining packages.

## 7. Inspect the SLAKE dataset

Run this before full inference to confirm the dataset schema:

```bash
python scripts/inspect_slake.py \
  --dataset_repo mdwiratathya/SLAKE-vqa-english \
  --split test
```

This prints:

- available split names
- dataset columns
- the first raw sample
- extracted image/question/answer/sample_id/image_id fields

The default dataset repository is:

```text
mdwiratathya/SLAKE-vqa-english
```

Other candidate repositories can be tried with `--dataset_repo`, for example:

```bash
python scripts/inspect_slake.py --dataset_repo BoKelvin/SLAKE --split test
python scripts/inspect_slake.py --dataset_repo Voxel51/SLAKE --split test
```

## 8. Run inference

Short answer setting:

```bash
python scripts/run_inference.py \
  --dataset_repo mdwiratathya/SLAKE-vqa-english \
  --split test \
  --conditions original black text_only \
  --num_samples 1000 \
  --max_new_tokens 16 \
  --image_size 512
```

Longer answer setting:

```bash
python scripts/run_inference.py \
  --dataset_repo mdwiratathya/SLAKE-vqa-english \
  --split test \
  --conditions original black text_only \
  --num_samples 1000 \
  --max_new_tokens 64 \
  --image_size 512
```

Dry run without loading the model:

```bash
python scripts/run_inference.py \
  --dataset_repo mdwiratathya/SLAKE-vqa-english \
  --split test \
  --conditions original black text_only \
  --num_samples 3 \
  --dry_run
```

Resume an interrupted run:

```bash
python scripts/run_inference.py \
  --dataset_repo mdwiratathya/SLAKE-vqa-english \
  --split test \
  --conditions original black text_only \
  --num_samples 1000 \
  --max_new_tokens 16 \
  --image_size 512 \
  --resume
```

Save resized debug images:

```bash
python scripts/run_inference.py \
  --dataset_repo mdwiratathya/SLAKE-vqa-english \
  --split test \
  --conditions original black \
  --num_samples 10 \
  --max_new_tokens 16 \
  --image_size 512 \
  --save_images
```

## 9. Summarize outputs

After inference:

```bash
python scripts/summarize_outputs.py
```

Or summarize one file:

```bash
python scripts/summarize_outputs.py \
  --input outputs/raw/qwen_slake_mdwiratathya_SLAKE_vqa_english_test_mnt16.jsonl
```

The summary does not score correctness. It reports:

- number of rows per condition
- number of errors
- average prediction length
- frequent answers
- ratio of unknown-like answers

Unknown-like keywords:

```text
unknown
cannot determine
can't determine
not enough information
unable to tell
cannot tell
not visible
```

## 10. Result file locations

Raw JSONL outputs:

```text
outputs/raw/
```

Summary tables:

```text
outputs/tables/output_summary.csv
outputs/tables/output_summary.md
```

Optional debug images:

```text
outputs/images/
```

Each JSONL row contains fields such as:

```text
model_name
dataset
condition
sample_id
question
gt_answer
pred_answer
image_size
max_new_tokens
do_sample
temperature
random_seed
split
image_id
raw_sample_keys
error
```

## Notes

- Batch size is intentionally fixed to 1 for simplicity and easier debugging.
- `do_sample=False` is used for greedy decoding.
- `temperature=0.0` is recorded in the output, but temperature is not passed to `generate()` unless sampling is enabled.
- The text-only condition sends only the question text to the model.
- The image conditions resize images to `image_size x image_size` before inference.
