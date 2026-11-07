# Decision 0008: KAIST Acoustic v6 Subset Integration

- Status: Accepted
- Proposed: 2026-11-06
- Accepted: 2026-11-06
- Scope: fourth dataset integration (different sensor modality); no kernel changes

## Decision

Integrate the acoustic subset of the CC BY 4.0 KAIST rotating-machine dataset
(DOI 10.17632/ztmf3m7h5x.6) as `kaist-rotating-acoustic-v6`: five Test.Lab MAT
recordings (51.2 kHz, Pascal), checksum-pinned with the archive hash published
by Mendeley, streamed through the generic signal contracts.

## Rationale

1. RQ4 requires sensor-modality heterogeneity; acoustic pressure is a first
   modality beyond vibration, with unit-aware channel metadata (`acoustic_pa`,
   `Pa`).
2. The Mendeley API publishes per-archive SHA-256 hashes, enabling end-to-end
   INV-1 verification from the source to each extracted file.
3. The Test.Lab structure is parsed generically (y-values plus x-axis
   increment), and the implied rate is cross-checked per file.

## Boundaries

- No task manifest is frozen for the **acoustic** subset: that archive covers a
  single load (0 N·m), so no three-way disjoint partition with complete label
  coverage exists. Freezing a task would require pretending a condition shift
  that the data does not contain.
- The **vibration** archive was integrated on 2026-11-07 (45 checksum-pinned
  recordings, see its data card) with the frozen five-class cross-load task
  `kaist-vibration-load-0-2-to-4-v1`; the upstream "Unbalalnce" misspelling in
  the 2 N·m files is normalized for labels while original filenames stay
  pinned. Full end-to-end training awaits a ≥16 GB machine.
- Current/temperature TDMS (1.5 GB) remains a future integration.
