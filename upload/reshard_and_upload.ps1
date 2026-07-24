<#
  reshard_and_upload.ps1
  Fix for: "Your push was rejected ... up to 10000 files ... Offending directories: /frames/"

  HF limits every directory in a repo to 10,000 files. This script moves your flat
  frames into subfolders (frames/shard_000, shard_001, ...), each well under the
  limit, then uploads them with the RESUMABLE large-folder uploader.

  Nothing here needs a GPU; it only moves files and uploads, so it runs fine on your
  laptop. If the upload drops, just re-run the SAME command — it resumes.

  USAGE (PowerShell, from anywhere):
    ./reshard_and_upload.ps1                          # uses the defaults below
    ./reshard_and_upload.ps1 -ShardSize 5000 -Copy    # copy instead of move
#>

param(
  [string]$SourceFrames = "$HOME\Desktop\frames1\frames",   # your current flat folder
  [string]$Staging      = "$HOME\Desktop\LLMSTU_hf",        # clean upload root (created)
  [string]$Repo         = "WaelK/LLMSTU",
  [int]   $ShardSize    = 5000,                             # <= 10000; 5000 is safe
  [int]   $NumWorkers   = 8,
  [switch]$Copy                                             # copy files instead of moving
)

$ErrorActionPreference = "Stop"
$framesOut = Join-Path $Staging "frames"

Write-Host "==> Source : $SourceFrames"
Write-Host "==> Staging: $framesOut  (repo will get frames/shard_XXX/...)"
$mode = if ($Copy) { "COPY" } else { "MOVE" }
Write-Host "==> Repo   : $Repo   ShardSize=$ShardSize   Mode=$mode"

if (-not (Test-Path $SourceFrames)) { throw "Source folder not found: $SourceFrames" }
New-Item -ItemType Directory -Force -Path $framesOut | Out-Null

# --- 1. reshard -------------------------------------------------------------
# Only loose files in the top of the source get sharded (skip anything already
# organised into shard_* subfolders, so re-running is safe).
$files = Get-ChildItem -LiteralPath $SourceFrames -File
$total = $files.Count
if ($total -eq 0) {
  Write-Host "No loose files in source (already sharded?). Skipping reshard."
} else {
  Write-Host "Resharding $total files into shards of $ShardSize ..."
  $i = 0
  foreach ($f in $files) {
    $shard = "shard_{0:D3}" -f [int][math]::Floor($i / $ShardSize)
    $dir   = Join-Path $framesOut $shard
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    $dest = Join-Path $dir $f.Name
    if ($Copy) { Copy-Item -LiteralPath $f.FullName -Destination $dest }
    else       { Move-Item -LiteralPath $f.FullName -Destination $dest }
    $i++
    if ($i % 2000 -eq 0) { Write-Host ("  {0}/{1}" -f $i, $total) }
  }
  Write-Host "Reshard done -> $framesOut"
}

$shardCount = (Get-ChildItem -LiteralPath $framesOut -Directory).Count
Write-Host "Shards ready: $shardCount folders"

# --- 2. upload (resumable) --------------------------------------------------
# Uploads the Staging root so the repo layout becomes frames/shard_XXX/*.jpg.
# Re-run this same command to resume if it stops.
Write-Host "Starting upload (resumable). This can take a while for ~35GB..."
hf upload-large-folder $Repo $Staging --repo-type=dataset --num-workers=$NumWorkers

Write-Host ""
Write-Host "Done. Verify at: https://huggingface.co/datasets/$Repo/tree/main/frames"
