# Second Dataset Source and License Review — 2026-07-30

## Purpose

This review selects a second independent source to test the v1.1 dataset contracts. Selection
requires an official landing page, explicit access terms, stable version identity, meaningful
condition domains, and enough file metadata to audit downloads. Popularity alone is not a
selection criterion.

## Candidate comparison

| Candidate | Official rights | Useful domains | Auditability | Decision |
|---|---|---|---|---|
| Mehran triaxial induction-motor bearing v2 | CC BY 4.0 | 100/200/300 W, inner/outer race, six defect sizes, three axes | Mendeley API exposes 38 files, byte sizes, SHA-256, and file IDs | Preferred |
| University of Ottawa UODS-VAFDC | CC BY 4.0 | 20 bearings, vibration, acoustic, speed, load, three health stages | Versioned Mendeley source; most labels use one load | Retain for a later multimodal/device task |
| Paderborn Bearing DataCenter | CC BY-NC 4.0 | vibration, motor current, multiple operating conditions and damages | Official university source | Not preferred because the non-commercial restriction narrows reuse |
| Ferrara artificial outer-race dataset | CC BY 4.0 | load, shaft speed, defect size | Versioned Mendeley source, but signals are bundled in a RAR archive | Not preferred for the first adapter test |

Official sources:

- [Mehran dataset](https://data.mendeley.com/datasets/fm6xzxnf36/2), DOI
  `10.17632/fm6xzxnf36.2`;
- [Ottawa dataset](https://data.mendeley.com/datasets/y2px5tg92h/5);
- [Paderborn Bearing DataCenter](https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter);
- [Ferrara dataset](https://data.mendeley.com/datasets/8wdzm5gwng/1).

## Mehran v2 structure

The latest public API record reports 159,324,234 bytes across 38 CSV files under CC BY 4.0.
Thirty-six files form a complete factorial grid:

- fault location: inner race or outer race;
- defect size: 0.7, 0.9, 1.1, 1.3, 1.5, or 1.7 mm;
- load: 100, 200, or 300 W.

Two additional healthy files describe operation with and without a pulley but do not encode a
100/200/300 W load. They cannot be inserted into a cross-load classification task without
inventing metadata and are therefore excluded from the proposed first task. The exclusion is a
protocol limitation, not evidence that healthy operation is unimportant.

The source describes 10 kHz triaxial acquisition. A downloaded official sample
(`0.7inner-100watt-67V2Iv.csv`) contained a header plus 143,348 rows with time stamp, X, Y, and
Z columns. Its observed size was 4,520,076 bytes and its observed SHA-256 was
`3b1f1ebd2499cd75a4443455afcc4101bd64136b02ea7c1ba3caed3c86780724`, exactly matching
the public API record.

## Proposed first task

Use the 36 complete fault recordings for two-class fault-location diagnosis:

- labels: `inner_race` and `outer_race`;
- held-out domain: `load_w`;
- evaluation group: `defect_size_mm`;
- train/validation/test: 100/200/300 W;
- reverse task: 300/200/100 W;
- recording identity: the exact upstream CSV file;
- raw data: downloaded and checksum verified, never committed.

This task validates a second institution, CSV parsing, triaxial signals, non-integer domain
metadata, and adapter extensibility. It does not establish healthy-versus-fault detection,
device generalization, natural degradation, or field validity.
