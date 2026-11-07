# Gearbox Two-Stage v1 — Data Card

- Dataset: "Gearbox Fault Diagnosis Data" (two-stage SpectraQuest gearbox simulator)
- Source: [data.gov catalogue entry](https://catalog.data.gov/dataset/gearbox-fault-diagnosis-data),
  archive `https://data.openei.org/files/623/gearboxdata.zip`
  (SHA-256 pinned in [`registry/gearbox_v1.json`](../../registry/gearbox_v1.json))
- Licence: CC BY 4.0, stated on the catalogue page; publisher contact Yogesh Pandya (PIEMR)
- Version: v1 integration, 20 recordings selected from the 20-file archive (all files)

## Contents

- Two gear conditions: `healthy` (files `h30hz*`) and `broken_tooth` (`b30hz*`).
- Ten load levels per condition (0–90 % in steps of 10) at a 30 Hz shaft speed.
- Four accelerometers per recording (tab-separated columns), 88k–114k samples per file.
- Related publication: "Fault detection of a two-stage gearbox with artificially
  induced gear crack using vibration monitoring" (see catalogue page).

## Known limitations

- **The source states no sampling rate.** The adapter records `sample_rate_hz=None`;
  secondary literature cites values around 10 kHz, but the framework refuses to
  assume one, so `signal_features` preprocessing fails closed on this dataset.
  Only sample-index-based preprocessing (`per_window_zscore`) is available.
- Rows are tab separated with a trailing tab; the loader reads columns 0–3 only.
- Row counts in the registry follow the loader's own parse (files end with blank
  lines; naive line counts overstate them by up to two).
- One recording per (condition, load) cell: recording-level statistics have
  20 units total.

## Tasks

- [`gearbox-load-low-to-high-v1`](../../registry/tasks/gearbox_load_low_to_high_v1.json):
  train loads 0–30 %, validate 40–50 %, test 60–90 %; both labels present in every
  partition; evaluation groups by gear condition.

## Acquisition

```bash
uv run edgefault dataset fetch --manifest registry/gearbox_v1.json
```

Downloads the archive, verifies its SHA-256, extracts the twenty files, and
re-verifies each against the registry before use.
