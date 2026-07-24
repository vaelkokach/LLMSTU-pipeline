# Final run on a ~95 GB GPU (H100 NVL / GH200 class)

`config.yaml` now ships tuned for a **single ~95 GB card**. This is a much roomier
setup than the 40 GB A100, so the strategy changes: run the model at **full bf16**
(no quantization) and spend the spare VRAM on **throughput**.

## What changed vs the 40 GB profile

| knob | 40 GB A100 | 95 GB card | why |
|---|---|---|---|
| `load_in_4bit` | true | **false** | 27B bf16 (~54 GB) now fits with ~40 GB to spare; bf16 is higher quality and avoids 4-bit dequant overhead |
| `batch_size` | 8–16 | **32** (push 48–64) | ~40 GB headroom for activations/KV |
| `attn_implementation` | sdpa | **flash_attention_2** | faster + less memory; big card justifies installing flash-attn |
| `crop.out_size` | 448 | **512** | more facial detail; you have the compute |
| `max_pixels` | 451k–602k | **803k** (1024·28·28) | ≥ 512px crop area, so 512 crops aren't downscaled |
| `vllm_gpu_mem_util` | 0.85 | **0.92** | claim most of the 95 GB |
| `vllm_max_num_seqs` | 32–64 | **128** | high concurrency |

## Memory budget (27B bf16)

```
weights            ~54 GB
KV cache + acts    scales with batch_size × seq_len × layers
image features     small (crops are 512px)
--------------------------------------------
target             keep peak under ~90 GB (leave headroom)
```

Start at `batch_size: 32`, watch `nvidia-smi`, and raise toward **48–64** until you're
around 85–90 GB peak, then back off one step. If you ever OOM, drop `batch_size` first,
then `max_pixels`.

## Two ways to run

### A) transformers (simple, already the default)
Good throughput with big batches and flash-attn. Just run the pipeline notebook /
scripts with the shipped config.

### B) vLLM (fastest — recommended for the full 104k-frame run)
Continuous batching keeps the big card saturated. Switch:

```yaml
caption:
  backend: vllm
  vllm_gpu_mem_util: 0.92
  vllm_max_num_seqs: 128
```
```bash
pip install "vllm>=0.11.0"
```
On a 95 GB card vLLM can hold the 27B weights **and** a large KV cache, so it typically
beats transformers by a wide margin on a run this size. Do a `--limit 500` pass first to
confirm label quality and measure crops/sec.

## Even more headroom? Options

You have room to trade some of that VRAM for **quality** instead of pure speed:

- **Bigger model**: a larger Qwen3.5 dense or MoE variant fits in bf16 on 95 GB (or in
  4-bit if you want the very largest). Set `caption.model_id` and re-check memory.
- **Higher-detail crops**: bump `crop.out_size` to 640–768 and `max_pixels` to match
  (e.g. `1280*28*28`) for small/far students — verify it actually helps on the tuning
  report before paying for it across the whole set.

Pick one lever at a time and confirm on `scripts/06_tune_captions.py` output.

## Flash-attention install

```bash
pip install flash-attn --no-build-isolation
```
If it fails to build in your environment, set `attn_implementation: sdpa` — everything
else stays the same, you just lose a bit of speed.

## Still resumable

Cropping and captioning both append to JSONL and skip finished work, so even on a fast
card you can stop/restart safely. For the full run, the shard-by-shard loop in
`docs/COLAB_A100.md` §4 applies here too — it's just faster.

## One-line sanity gate

```bash
python scripts/02_caption.py --limit 500 --crops ./work/crops --out ./work/pl_smoke.jsonl
```
Read the labels, note crops/sec from the progress print, extrapolate to your full crop
count, then launch the real run.
