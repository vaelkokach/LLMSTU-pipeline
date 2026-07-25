# LLMSTU dataset enhancement

Turn raw classroom frames (`WaelK/LLMSTU`, ~104k JPGs under `frames/`) into a
student-level dataset:

1. **Detect + crop** every student from each frame → wide face / upper-body crops
   (one crop = one training sample).
2. **Pseudo-label** each crop with **Qwen3.5** (natively multimodal — the model
   weights run **on your GPU**, no inference API is called) into a structured
   engagement/attention schema (JSON per crop).
3. **Golden eval set** — sample a balanced subset and correct it by hand with the
   included browser labeling tool.
4. **Package + push** crops + pseudo-labels + golden set to a new HF dataset.

Designed to run on a **cloud GPU (Colab or Hugging Face)** — nothing needs your
local machine. The same `llmstu/` package backs the notebook and the CLI scripts.

---

## Quickest path — Colab

Open `notebooks/LLMSTU_pipeline_colab.ipynb` in Colab (GPU runtime), set your HF
token in the first cell, and Run All. It clones this repo, installs deps, and runs
the four stages. Use a **T4/A100** runtime; pick model size by GPU (see below).

## Run it as scripts (HF Jobs, RunPod, any GPU box)

```bash
pip install -r requirements.txt
export HF_TOKEN=hf_xxx                      # write token

# 1) detect + crop students  (downloads frames from HF if --frames omitted)
python scripts/01_crop.py --out ./work/crops

# 1b) drop near-duplicate crops (1-fps footage) BEFORE captioning — big cost saver
python scripts/09_dedup.py --manifest ./work/crops_manifest.jsonl --crops ./work/crops \
    --out ./work/crops_manifest_dedup.jsonl

# 2) pseudo-label the DEDUPED crops with Qwen3.5
python scripts/02_caption.py --manifest ./work/crops_manifest_dedup.jsonl \
    --crops ./work/crops --out ./work/pseudo_labels.jsonl

# 3) build the golden-eval sample (stratified + hard subset) + labeling payload
python scripts/03_sample_eval.py --labels ./work/pseudo_labels.jsonl --crops ./work/crops \
    --n 1500 --stratify-fields engagement_level,activity,gaze_direction \
    --min-per-class 40 --hard-frac 0.2 --embed

# 4) score the pseudo-labels against your hand-corrected golden.json
python scripts/07_eval_golden.py --golden labeling/golden.json

# 5) package + push crops (sharded) + captions + golden set to a new dataset repo
python scripts/04_upload.py --repo WaelK/LLMSTU-crops
```

**Tip — validate before the full run:** caption just 1–2 shards, hand-label ~200
crops, and score them first. See `docs/EARLY_GOLDEN_CHECK.md`.

**Near-duplicate removal (1-fps footage).** Because frames are 1/second, a still
student produces many near-identical crops. `scripts/09_dedup.py` groups crops by
(video, seat region), orders them in time, and keeps a crop only when it differs
enough (perceptual dHash) from the last kept one — collapsing static runs while
keeping frames where something changes. Run it between crop and caption; it can cut
the crop count (and caption cost) substantially. Tune `dedup.hamming_threshold` in
`config.yaml` (4 = gentle, 6 = default, 8–10 = aggressive).

**Golden set (step 3).** Size is driven by precision + category coverage, not the
million-crop pool — 1,500–2,000 gives ~±2.5% margin. `03_sample_eval.py` builds a
**representative** set (stratified over several fields with a per-class floor so rare
categories are covered) plus a **hard** subset (lowest `model_confidence`, where
errors concentrate). `07_eval_golden.py` reports accuracy for each group separately.

### What gets uploaded (step 5)

The crops are saved as images and pushed **sharded** (`crops/shard_000/…`, ≤5000
files each, so the 10k-per-directory limit is never hit), with a
`crops/metadata.jsonl` that pairs **every crop image with its caption + label
fields** (HF `imagefolder` convention — the dataset viewer shows image + caption
together, and `load_dataset("imagefolder", ...)` returns them joined). The repo also
gets `pseudo_labels.parquet` (full table) and `golden_eval.json`.

## Hand-labeling the golden set

After step 3, open the tool:

```bash
cd labeling && python -m http.server 8000   # then visit http://localhost:8000
```

Each crop is shown with Qwen3-VL's prediction pre-filled. Correct the fields
(changed fields are outlined amber), press **v** to mark a crop as agreeing with
the model, and use **←/→** to move. Click **Export golden.json** when done and drop
that file back next to the crops before running step 4. If you used `--embed` in
step 3, `data.json` is fully self-contained and you can even open `index.html`
by double-clicking (no server needed).

---

## What the labels look like

Structured, engagement/attention focused (full definition in `llmstu/schema.py`):

| field | values |
|---|---|
| `activity` | listening, writing_notes, reading, using_phone, using_laptop, talking_to_peer, raising_hand, head_down_sleeping, looking_away, eating_drinking, other |
| `gaze_direction` | teacher_or_board, own_desk, peer, phone, laptop, away_or_window, down, unknown |
| `attention_target` | instruction, own_work, peer, device, distracted, unknown |
| `engagement_level` | engaged, partially_engaged, disengaged, unknown |
| `posture` | upright, leaning_forward, leaning_back, slumped, head_down, turned_away |
| `hand_state` | writing, on_phone, on_desk_idle, raised, on_face, gesturing, unknown |
| `phone_visible` / `laptop_visible` / `talking` / `occluded` | true / false |
| `caption` | one-sentence summary |
| `model_confidence` | 0.0–1.0 (model self-rating, used to triage what to hand-check) |

## Tune first, then run (recommended workflow)

1. **Tune** with `notebooks/LLMSTU_tuning_colab.ipynb` (or `scripts/05_tune_crops.py`
   and `scripts/06_tune_captions.py`). Edit `experiments.yaml` to sweep crop configs and
   caption prompts; the scripts render HTML reports so you can compare crop
   tightness/coverage and the JSON each prompt produces (with per-field agreement).
2. **Lock in** the winners in `config.yaml` (`crop:`, `caption.model_id`,
   `caption.prompt_name`) and tweak `llmstu/schema.py` if you want to change label fields.
3. **Full run** with `notebooks/LLMSTU_pipeline_colab.ipynb` — see
   **`docs/COLAB_A100.md`** for max-efficiency A100 settings.

Prompt variants live in `llmstu/prompts.py` (`v1_structured`, `v2_concise`,
`v3_reasoned`, `v4_fewshot`); add your own and reference it by name.

**Two separate tuning axes.** The *fields themselves* are your design in
`llmstu/schema.py` — Qwen only fills values, it never chooses fields. So schema
(which fields/categories exist) and prompt wording are different knobs:

- `scripts/06_tune_captions.py` — A/B prompt **wording** (fields held constant).
- `scripts/08_tune_schema.py` — A/B **field sets** (`schema.SCHEMAS`: `full`,
  `no_attention_target`, `gaze_merged`, `lean`), reporting each enum field's
  "concrete-rate" (how often the model answered something other than `unknown`).
  Low-agreement fields from wording tuning (e.g. `gaze_direction`,
  `attention_target`) are the ones to test here. Set the winner via
  `caption.schema_name` in `config.yaml`.

Decide the schema first, then tune wording under it.

## Model: Qwen3.5 (runs on the GPU, not an API)

`caption.model_id` defaults to `Qwen/Qwen3.5-27B`. Qwen3.5 is natively multimodal, so
the base model reads images directly — there's no `-VL` suffix. The weights are
downloaded and executed **locally on your GPU** through
`AutoModelForImageTextToText`; nothing is sent to a hosted inference endpoint.

`config.yaml` ships **tuned for a single ~95 GB GPU** (H100 NVL / GH200 class): 27B in
full **bf16** (no 4-bit), `batch_size: 32`, flash-attention, 512px crops. See
**`docs/GPU_95GB.md`**.

| your GPU | setup | notes |
|---|---|---|
| ~95 GB (H100 NVL / GH200) | `Qwen/Qwen3.5-27B` bf16, batch 32–64 | **shipped default** — see `docs/GPU_95GB.md` |
| 40 GB A100 | `Qwen/Qwen3.5-27B` + `load_in_4bit: true`, smaller batch | see `docs/COLAB_A100.md` |
| smaller card | a smaller Qwen3.5 size, 4-bit | faster, slightly lower quality |

For the fastest full pass, switch `caption.backend: vllm` (install `vllm>=0.11.0`) —
on 95 GB it holds the weights plus a large KV cache and clearly beats transformers.
Getting the repo into Colab (GitHub optional): **`docs/GET_INTO_COLAB.md`**.

## Config

Everything lives in `config.yaml`; the knobs you'll touch most:

- `crop.crop_mode` — `head_wide` (default, wide face crop), `upper_body`, or `person`.
- `crop.margin_frac` / `crop.head_frac` — how much context around the face.
- `crop.min_side_px` — drop tiny far-away students.
- `crop.detector_model` — `yolo11n/s/m/l/x.pt` (recall vs speed).
- `eval.stratify_by` / `eval.n_samples` — balance + size of the golden set.

Env overrides: `LLMSTU_MODEL_ID`, `LLMSTU_BACKEND`, `LLMSTU_SOURCE_REPO`,
`LLMSTU_CROPS_REPO`, `LLMSTU_WORKDIR`, `HF_TOKEN`.

## Notes / best practices baked in

- Detector runs at `imgsz=1280` — wide-angle rooms have small students; downscaling
  loses them.
- Crops are resumable (JSONL manifest) — a dropped 100k-frame run continues where it
  stopped. Captioning is resumable too.
- Faces get extra top padding so hair / hand-on-face gestures aren't cut off.
- The model is told **not** to infer identity, age, gender, or ethnicity — labels are
  behavioral only.
- Pseudo-labels are never treated as ground truth: the golden set is the eval anchor,
  and `model_confidence` tells you which crops to prioritize for human review.
