# 224-GPU PP14 / CP2 / DP8 / EP8 extrapolation

This case is a work-conserving extrapolation from the frozen 256-GPU
iteration-55 trace. Fabric measurements now validate the aligned iteration-55
point and all 20 profiled iterations. The point error is within 3%, while the
20-iteration temporal P90 error fails the 5% gate.

Build the 120-scenario stage-template uncertainty sweep:

```bash
python case_224gpu_pp14_cp2_a2a/scripts/build_pp14_extrapolation_v1.py
```

The main outputs are written to `results/pp14_extrapolation_v1/`:

- `PP14_EXTRAPOLATION_V1.md`: assumptions, uncertainty range, and boundaries;
- `scenario_sweep.csv`: all `C(16,2)=120` work-conserving mappings;
- `source_stage_profile.csv` and `stage_omission_sensitivity.csv`: explain the
  endpoint-driven upper tail of the sweep;
- `summary.json`: machine-readable source/target metrics and quantiles;
- `validation.json`: structural checks and predictive-validation link;
- `fabric_measurement_validation/FABRIC_MEASUREMENT_VALIDATION.md`: measured
  224-GPU Step/MFU reconciliation and temporal-stability result;
- `fabric_measurement_validation/fabric_validation_by_iteration.csv`: all 20
  profiled iterations;
- `legal_224gpu_strategies.csv`: other factor-compatible PP/CP/DP candidates and
  their trace-identifiability class.

Re-run the measurement validation against the sibling Fabric repository:

```bash
python case_224gpu_pp14_cp2_a2a/scripts/validate_against_fabric_measurement.py
```

The v1 timing DAG still conserves the source 60-layer, four-microbatch work.
The main MFU numerator is corrected to the measured target workload contract
(52 layers, GBS48, three microbatches); v2 must also replace the timing DAG
with that target schedule.
