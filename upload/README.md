# Fixing the "too many files per directory" upload

Your first upload failed with:

> Your push was rejected because it contains too many files per directory. Each
> directory in your git repo can only contain up to 10000 files. Offending
> directories: /frames/

That is a **hard Hugging Face limit: max 10,000 files per directory**. All 104,069
images went into a single `frames/` folder, so it can never commit. The fix is to
split them into subfolders (`frames/shard_000 … shard_020`, ~5,000 each) and upload
with the resumable large-folder uploader.

## Step 0 — clear the half-uploaded folder (one time)

Your repo already has ~10,000 loose files committed under `frames/`. Remove them so
the new sharded layout is clean (this does **not** delete the repo or your local
files):

```powershell
python -c "from huggingface_hub import HfApi; HfApi().delete_folder(path_in_repo='frames', repo_id='WaelK/LLMSTU', repo_type='dataset')"
```

(If you'd rather start completely fresh instead:
`python -c "from huggingface_hub import delete_repo, create_repo; delete_repo('WaelK/LLMSTU', repo_type='dataset'); create_repo('WaelK/LLMSTU', repo_type='dataset', private=True)"`)

## Step 1 — reshard + upload

```powershell
cd $HOME\Desktop\frames1\LLMSTU-pipeline\upload
./reshard_and_upload.ps1
```

What it does:
1. Moves your flat images from `Desktop\frames1\frames` into
   `Desktop\LLMSTU_hf\frames\shard_000 … shard_020` (5,000 per shard, same drive so
   it's instant). Pass `-Copy` if you want to keep the originals in place.
2. Runs `hf upload-large-folder WaelK/LLMSTU <staging> --repo-type=dataset` — this is
   **resumable and parallel**. If it drops, re-run the exact same script and it
   continues where it left off.

Result in the repo: `frames/shard_000/…jpg`, `frames/shard_001/…`, etc.

## Why not just retry the old command?

`hf upload … .` puts everything in one directory again → same rejection.
`upload-large-folder` also won't help without sharding, because the 10k limit is a
property of the **repo directory**, not the CLI. Sharding is required either way.

## Note for the enhancement pipeline

The pipeline already reads frames recursively (`frames_glob: "**/*.jpg"` and it
downloads `frames/**`), so the new `frames/shard_XXX/` layout works with
`scripts/01_crop.py` and the Colab notebook **with no changes**.

## Alternative (fewer, bigger files — best practice at scale)

If you'd rather not manage 104k small files at all, pack them into ~21 parquet or
WebDataset tar shards instead. That uploads ~21 files total, is faster, and the HF
dataset viewer renders it natively. Ask and I'll add a `pack_to_parquet.py` that
does this and a loader for the pipeline.
