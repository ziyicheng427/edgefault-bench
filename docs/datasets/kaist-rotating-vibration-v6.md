# KAIST Rotating-Machine Vibration v6 — Data Card

- Dataset: same source as the acoustic subset
  ([Mendeley](https://data.mendeley.com/datasets/ztmf3m7h5x/6), DOI 10.17632/ztmf3m7h5x.6,
  CC BY 4.0); this card covers the **vibration archive**
  (`vibration.zip`, 2.66 GB, SHA-256 `df73fa80…33e4659` pinned in
  [`registry/kaist_vibration_v6.json`](../../registry/kaist_vibration_v6.json))
- 45 recordings: three loads (0/2/4 N·m) × fifteen conditions
  (Normal; BPFI and BPFO at three severities; misalignment at three levels;
  unbalance at five masses), four acceleration channels (housing A/B, x/y, g)
  at 25.6 kHz, durations 60–300 s per file (3.4 GB extracted).

## Known source quirks (handled explicitly)

- The 2 N·m unbalance files carry the upstream misspelling `Unbalalnce`; the
  adapter normalizes the label to `unbalance` while registry entries pin the
  original filename and hash.
- Per-file durations differ (Normal 7.68 M samples, bearing files 1.536 M,
  others 3.072 M); the registry records each file's sample count and the
  kernel cross-checks it on every open.

## Tasks

- [`kaist-vibration-load-0-2-to-4-v1`](../../registry/tasks/kaist_vibration_load_0_2_to_4_v1.json):
  five-class machine-condition classification (normal/bpfi/bpfo/misalign/
  unbalance), train 0 N·m, select 2 N·m, test unseen 4 N·m — the first frozen
  task on this dataset family and the first task on a non-bearing modality
  mix (bearing, alignment, and balance faults together).

## Verification status and boundary

- Registry, adapter, plugin discovery, fail-closed task audit (15/15/15
  recordings), and checksum-verified streaming were validated against the real
  archive on 2026-11-07, including the misspelled files.
- **Full end-to-end training on this task is not yet executed**: extracting
  all 45 recordings at the frozen window (4096/4096) materializes ~124 M
  samples × 4 channels (≈4 GB float64 plus training copies), which exceeds the
  8 GB development machine. A ≥16 GB machine (or the planned streaming/
  Parquet path) is required; the run remains open on the roadmap.

## Acquisition

```bash
uv run edgefault dataset fetch --manifest registry/kaist_vibration_v6.json
```

Requires ~6 GB free disk (archive plus extraction); every file is verified
against the registry before use.
