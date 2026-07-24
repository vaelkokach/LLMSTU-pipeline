# Running on Hugging Face

Two options, both keep compute + data on HF.

## A) HF Jobs (batch, recommended for the full run)

`hf jobs` runs a container on HF GPUs. From your laptop (only the CLI runs locally,
the work runs on HF):

```bash
pip install -U huggingface_hub
hf auth login          # write token

hf jobs run \
  --flavor a100-large \
  --secret HF_TOKEN=$HF_TOKEN \
  pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime \
  bash -c '
    pip install -q ultralytics "transformers>=4.57.0" accelerate huggingface_hub pandas pyarrow pyyaml &&
    git clone https://huggingface.co/datasets/WaelK/LLMSTU-pipeline-code /code 2>/dev/null || true &&
    git clone https://github.com/<you>/LLMSTU-pipeline /code &&
    cd /code &&
    python scripts/01_crop.py --out ./work/crops &&
    python scripts/02_caption.py --crops ./work/crops --out ./work/pseudo_labels.jsonl &&
    python scripts/04_upload.py --repo WaelK/LLMSTU-crops
  '
```

Pick `--flavor` by model size: `a10g-large` for 4B/8B, `a100-large` for 30B-A3B.
Push this repo to a GitHub repo (or an HF `model`/`dataset` repo) first so the job
can clone it.

## B) HF Space (interactive / smaller runs)

Create a **GPU Space** (Gradio or "blank"), add this repo, set `HF_TOKEN` as a Space
secret, and run the scripts from the Space terminal or a small Gradio wrapper. Good
for iterating on the crop/caption settings on a few hundred frames before committing
to the full run.

## Tips

- Do a **dry run first**: `python scripts/02_caption.py --limit 200` to sanity-check
  the labels and cost before the full pass.
- Captioning 100k+ crops is the expensive step — use `caption.backend: vllm` and an
  80GB GPU for the full run.
- Everything is resumable; if a job times out, re-run the same command and it skips
  finished work.
