# KAIST Rotating-Machine Acoustic v6 — Data Card

- Dataset: "Vibration, Acoustic, Temperature, and Motor Current Dataset of Rotating
  Machine Under Varying Load Conditions for Fault Diagnosis"
  ([Mendeley](https://data.mendeley.com/datasets/ztmf3m7h5x/6), DOI 10.17632/ztmf3m7h5x.6,
  Jung et al., KAIST)
- Licence: CC BY 4.0 (Mendeley dataset page)
- Version: v6; this integration pins the **acoustic archive only**
  (`acoustic.zip`, SHA-256 pinned in
  [`registry/kaist_acoustic_v6.json`](../../registry/kaist_acoustic_v6.json))

## Contents

- Five 60-second acoustic recordings (Siemens Test.Lab MAT), single channel in
  Pascal at 51.2 kHz (3,072,000 samples each):
  `0Nm_Normal`, `0Nm_BPFI_03`, `0Nm_BPFI_10`, `0Nm_BPFO_03`, `0Nm_BPFO_10`.
- Labels: `normal`, `bpfi_03`, `bpfi_10`, `bpfo_03`, `bpfo_10`; domains
  `load_nm`, `fault_family` (`Normal`/`Inner`/`Outer`), `severity_mm`.

## Known limitations and boundary

- The acoustic archive contains only the 0 N·m load, so **no condition-held-out
  task is frozen for this dataset yet**: every task needs at least three disjoint
  domain values with all labels present per partition. Cross-load and
  cross-severity tasks become possible once the vibration archive (2.7 GB, three
  loads, four channels) is integrated.
- The temperature/motor-current archive is TDMS (1.5 GB) and is not integrated;
  multimodal channel mixing is future work.
- The adapter parses the Test.Lab structure (`Signal.y_values.values` with the
  x-axis increment) and cross-checks the implied 51.2 kHz rate per file.

## Acquisition

```bash
uv run edgefault dataset fetch --manifest registry/kaist_acoustic_v6.json
```

Downloads the Mendeley archive (hash-verified against the published value),
extracts the five MAT files, and re-verifies each against the registry.
