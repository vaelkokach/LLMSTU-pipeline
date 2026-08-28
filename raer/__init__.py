"""RAER external task — pseudo-labeling a third-party academic-emotion dataset
with the SAME labeler used for LLMSTU.

The labeling code is deliberately NOT duplicated here. `raer.labeler` imports
`llmstu.caption`, `llmstu.schema`, `llmstu.prompts` and `llmstu.sampling` from
the LLMSTU-pipeline checkout, so "identical pseudo-labeling" is guaranteed by
construction rather than by keeping two copies in sync.

This package adds only what LLMSTU has no equivalent of:
  * naming   — parse RAER filenames into clip / subject identity
  * audit    — pre-flight checks on a RAER copy (run first)
  * frames   — clip -> frame manifest, in LLMSTU's manifest format
  * aggregate— frame-level cue labels -> one verdict per clip
  * compare  — RAER label vs Qwen cues vs human adjudication
"""

__all__ = ["naming", "audit"]
