# Decision 0001: HUST v1 Protocol

- Status: Proposed for maintainer ratification
- Proposed: 2026-07-26
- Scope: EdgeFault-Bench v1.0 primary benchmark

## Decision

Use HUST bearing v3 as the first public source. Select the four labels available across all
five bearing devices and three loads: healthy (`N`), inner-race (`I`), outer-race (`O`), and
combined inner/outer (`IO`). Use non-overlapping 4,096-sample windows and the fixed seeds 17,
29, and 43.

Freeze three tasks:

1. train 0 W, validate 200 W, test 400 W;
2. train 400 W, validate 200 W, test 0 W;
3. train bearing devices 6204-6206, validate 6207, test 6208.

For load-held-out tasks, report worst-bearing Macro-F1. For the device-held-out task, report
worst-load Macro-F1.

## Rationale

- Mendeley records the dataset as CC BY 4.0 and exposes immutable v3 file checksums.
- The paper documents five bearing devices, three loads, and consistent file semantics.
- Excluding ball-related labels removes two author-documented missing 6204 cases and preserves
  identical class support across all domains.
- Device- and load-held-out tasks address different generalization failures.
- Non-overlapping windows and recording-level partition assignment prevent signal leakage.

## Alternatives considered

- KAIST acoustic v6 was verified as CC BY 4.0, but the 47 MB acoustic archive contains only
  0 Nm recordings and cannot support the primary cross-load task.
- Random within-recording splits were rejected because overlapping or near-identical windows
  can inflate apparent generalization.
- Including all seven HUST labels was rejected because ball-related cases are not complete for
  every bearing device.

## Consequences and limitations

- The v1 classification scope is narrower than the full HUST label set.
- Artificial faults and device-specific amplitude cues limit field generalization.
- The device-held-out task mixes geometry and defect-severity shift.
- Any later change requires a new task/decision version; existing results remain available.

## Ratification record

The maintainer should change the status to `Accepted`, add the review date, and state any
requested modifications only after personally reviewing this decision and the linked data
card. This record must not be marked accepted solely by automated tooling.

