# Final run on a Colab A100 — max efficiency

This is for the **final full run** (all shards, ~104k frames → several hundred k
crops) after you've locked crop/prompt/schema settings during tuning.

Colab's A100 is a **single 40GB** card. Everything below assumes that.

## 0. First, the honest math

Captioning is the bottleneck, not cropping. If cropping yields ~4 students/frame you
have ~400k crops. At ~5–8 crops/sec (27B, 4-bit, batched) that's **~15–22 GPU-hours**.
Colab sessions time out (often ~12–24h and can disconnect earlier), so the run **must
be resumable** — it is (JSONL append + skip-done), but plan for 2–3 sessions and/or
process shard-by-shard. For a single unattended pass, HF Jobs or RunPod on an 80GB
card is calmer; see `hf/HF_JOBS.md`.

## 1. Model + memory choices (pick one)

| Setup | Fits 40GB? | Speed | Quality |
|---|---|---|---|
| `Qwen/Qwen3.5-27B` + `load_in_4bit: true` | yes | good | high |
| `Qwen/Qwen3.5-27B` bf16 | **no** (needs ~54GB) | — | — |
| A smaller Qwen3.5 dense/MoE size | yes, bf16 | faster | slightly lower |

Recommended for a 40GB A100: **27B in 4-bit**. Set in `config.yaml`:

```yaml
caption:
  model_id: Qwen/Qwen3.5-27B
  load_in_4bit: true
  attn_implementation: flash_attention_2   # install flash-attn; else use sdpa
  dtype: bfloat16
  max_new_tokens: 256        # labels are short JSON; don't overspend tokens
  batch_size: 16             # raise until ~90% VRAM; drop if OOM
  max_pixels: 602112         # = 768*28*28; lower to 451584 (576*28*28) for more speed
  temperature: 0.0
```

## 2. The single biggest speed lever: `max_pixels`

Qwen packs the image into visual tokens proportional to resolution. Your crops are
small (448px), so cap `max_pixels` low — it cuts tokens/crop dramatically with almost
no quality loss on face/upper-body crops. Try `576*28*28` (451584) or even
`512*28*28` (401408) and confirm on the tuning report that labels hold up.

## 3. Throughput settings

- **Batching**: `batch_size` 16–32 for 4-bit 27B. Push it until you see ~90% of 40GB
  used (`nvidia-smi`), then back off one step.
- **flash_attention_2**: `pip install flash-attn --no-build-isolation` then set
  `attn_implementation: flash_attention_2`. Falls back to `sdpa` if unavailable.
- **Faster downloads**: `pip install hf_transfer` and
  `os.environ['HF_HUB_ENABLE_HF_TRANSFER']='1'` — matters when pulling 104k frames.
- **Don't re-decode**: crop once to disk, caption from disk (the pipeline already does
  this). Keep crops on the local Colab disk, not Drive (Drive I/O is slow).
- **vLLM for the very fastest pass**: `caption.backend: vllm` gives the best tokens/sec
  via continuous batching. It has a heavier setup and less flexible memory on 40GB, so
  it's optional — `transformers` + 4-bit is the reliable default.

## 4. Run it shard-by-shard (resumable, session-safe)

Because frames are sharded (`frames/shard_000…020`), caption one shard at a time so a
disconnect only loses the in-flight shard:

```python
from pathlib import Path
from llmstu import crop as crop_mod, caption as caption_mod, dataset_io as io
from llmstu.config import load
cfg = load('config.yaml')

for shard in sorted(Path('work/frames_dl/frames').glob('shard_*')):
    man = Path(f'work/manifests/{shard.name}.jsonl')
    crop_mod.run(io.iter_frames(shard, '*.jpg'), Path(f'work/crops/{shard.name}'),
                 cfg.crop, manifest_path=man)                       # resumable
    caption_mod.run(man, Path(f'work/crops/{shard.name}'),
                    Path('work/pseudo_labels.jsonl'), cfg.caption)  # resumable, appends
```

Re-running the loop after a disconnect skips finished frames/crops automatically.

## 5. Keep the session alive & checkpoint

- Push `work/pseudo_labels.jsonl` to your dataset repo every few shards (cheap insurance):
  `HfApi().upload_file(...)` or just re-run `scripts/04_upload.py` at the end.
- A100 runtimes idle-disconnect; keep the tab active. If it drops, reconnect, re-mount,
  re-run — nothing is lost.

## 6. Sanity gate before committing hours

Always run `--limit 200` (or the tuning notebook) first and read the report. Confirm
label quality and measure crops/sec, then extrapolate the full-run cost before you
start the long pass.
