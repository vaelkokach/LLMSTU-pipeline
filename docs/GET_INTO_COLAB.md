# Getting this repo into Colab (do I need GitHub?)

**Short answer: no, GitHub is not required — but it's the smoothest option.** The
notebook just needs the `llmstu/` package + scripts present in the Colab working
directory. Any of these gets them there.

## Option A — GitHub (recommended)

Cleanest and most repeatable. One-time push, then `git clone` in every Colab session.

```bash
# on your machine, inside LLMSTU-pipeline/
git init
git add .
git commit -m "LLMSTU enhancement pipeline"
git branch -M main
git remote add origin https://github.com/<you>/LLMSTU-pipeline.git
git push -u origin main
```

Then in the notebook's first cell set `REPO_URL = "https://github.com/<you>/LLMSTU-pipeline"`.
Private repo? Use a token in the URL: `https://<TOKEN>@github.com/<you>/LLMSTU-pipeline`.

## Option B — Hugging Face repo (no GitHub account needed)

You already use HF. Push the code to a `model`/`dataset` repo and clone it in Colab:

```bash
hf upload <you>/LLMSTU-pipeline-code . --repo-type=model   # from inside the folder
```
```python
!git clone https://huggingface.co/<you>/LLMSTU-pipeline-code
%cd LLMSTU-pipeline-code
```

## Option C — Upload the folder to Colab (zero hosting)

Zip the folder, drag it into the Colab file browser, unzip:

```python
!unzip -q LLMSTU-pipeline.zip
%cd LLMSTU-pipeline
```

Simplest for a one-off, but you re-upload each new session.

## Option D — Google Drive

Put the folder in Drive, mount it:

```python
from google.colab import drive; drive.mount('/content/drive')
%cd /content/drive/MyDrive/LLMSTU-pipeline
```

Handy because edits persist, but Drive file I/O is slow — keep the big `work/` outputs
on Colab's local disk (`/content/work`), not on Drive.

## Which to use

- Iterating on code and re-running often → **A (GitHub)** or **D (Drive)**.
- Just want to run it once → **C (upload zip)**.
- Want everything inside HF → **B**.

In all cases, set `HF_TOKEN` (Colab Secrets → add `HF_TOKEN`) so the notebook can pull
frames and push results.
