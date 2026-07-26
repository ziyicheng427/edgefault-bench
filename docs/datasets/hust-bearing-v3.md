# HUST Bearing v3 Data Card

## Source and license

- Dataset: *HUST bearing: a practical dataset for ball bearing fault diagnosis*.
- Repository version: Mendeley Data v3.
- Dataset DOI: [10.17632/cbv7jyx4p9.3](https://doi.org/10.17632/cbv7jyx4p9.3).
- Data paper: [10.1186/s13104-023-06400-4](https://doi.org/10.1186/s13104-023-06400-4).
- License recorded by Mendeley Data: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

EdgeFault-Bench does not redistribute the MAT files. The committed registry pins the
official file identifiers, byte sizes, SHA-256 checksums, and download URLs. Users must
retain attribution to Nguyen Duc Thuan and Hoang Si Hong and to the dataset source.

## Verified structure

The data paper reports 51,200 samples per second for 10 seconds at three loads: 0 W, 200 W,
and 400 W. Five bearing devices are included: 6204, 6205, 6206, 6207, and 6208. File names
encode the fault, device, and load; for example, `I402.mat` denotes an inner-race fault on
bearing 6204 at 200 W.

The v1 benchmark selects four labels present for every bearing and load:

- `N`: healthy;
- `I`: inner-race fault;
- `O`: outer-race fault;
- `IO`: combined inner- and outer-race fault.

This produces 60 pinned recordings. Excluding ball-related labels avoids the two missing
6204 cases documented by the authors and keeps the label support identical across domains.

## Windowing and leakage controls

- Use only the steady-state `data` variable; `ru_raw` is outside v1 scope.
- The 57 `N`/`I`/`O` recordings contain 512,000 steady-state samples. The three `IO400`,
  `IO402`, and `IO404` recordings contain 512,001; v1 deterministically drops their final
  sample before windowing so every recording has identical support.
- Use 4,096-sample windows with a 4,096-sample stride.
- Discard the incomplete tail rather than pad it.
- Center and scale each window independently.
- Treat the complete MAT file as the recording group. A recording can occur in exactly one
  task partition.
- Never estimate normalization statistics or choose hyperparameters from the test domain.

Each 512,000-sample recording yields 125 non-overlapping windows. The full selected subset
therefore contains 7,500 windows before robustness transformations.

## Frozen tasks

1. `hust-load-0-to-400-v1`: train at 0 W, validate at 200 W, test at 400 W.
2. `hust-load-400-to-0-v1`: train at 400 W, validate at 200 W, test at 0 W.
3. `hust-device-6204-6206-to-6208-v1`: train on 6204-6206, validate on 6207, test on 6208.

The first two tasks measure unseen-load reliability. The third measures transfer to a
different bearing size/device while retaining all three loads inside each device partition.
For the load-held-out tasks, worst-condition Macro-F1 is the worst bearing-device score.
For the device-held-out task, it is the worst load score on bearing 6208. This prevents the
worst-condition metric from collapsing to the single held-out domain identifier.

## Known limitations

- Faults are artificially introduced micro-cracks and do not cover natural degradation.
- Crack depth differs across bearings, so amplitude can expose device-specific cues.
- Environmental noise and mechanical imperfections are not represented in the source data.
- The device-held-out task can therefore measure both geometry and defect-severity shifts;
  it must not be interpreted as a pure bearing-size effect.
