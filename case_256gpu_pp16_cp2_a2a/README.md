# 256-GPU PP16 MFU case

The frozen case models iteration 55 from 256 ranks and 16 independent pipeline
lanes.  Each rank contributes four FWD and four BWD compute annotations.

## Versions

- `results/pp_dag_minimal`: FWD/BWD plus bidirectional PP network service.
- `results/pp_optimizer_dag_v1`: v0 frontier joined to DP/EDP RS and two AG rounds.
- `results/pp_optimizer_dag_v2`: backward network service and cluster software
  completion separated; current replay error is -0.1010% for ProfilerStep and
  +0.0954% relative for MFU.
- `results/pp_optimizer_dag_v3_oisa_mock`: preserves the v2 compute/software
  calibration and exercises the staggered-rank OISA FCT contract.  Its ×1.2
  network-tail scenario is a deterministic interface test, not a network
  prediction.
- `results/oisa_real_validation_faa3c78`: uses the real OISA/ns-3 binary for
  the release-offset smoke and one critical EDP-AG0 group, then replays that
  result through the complete MFU DAG.  It is an interface validation on the
  fixed 8-GPU topology, not a 256-GPU calibration.

The authoritative v2 explanation and validation are:

- `results/pp_optimizer_dag_v2/PP_OPTIMIZER_DAG_V2.md`
- `results/pp_optimizer_dag_v2/validation.json`
- `results/pp_optimizer_dag_v2/pp_software_completion_calibration.csv`

The v3 OISA boundary and slack outputs are:

- `results/pp_optimizer_dag_v3_oisa_mock/oisa_mock_requests.csv`
- `results/pp_optimizer_dag_v3_oisa_mock/oisa_mock_responses.csv`
- `results/pp_optimizer_dag_v3_oisa_mock/collective_slack_audit.csv`
- `results/pp_optimizer_dag_v3_oisa_mock/PP_OPTIMIZER_DAG_V3_OISA_MOCK.md`

The real-OISA validation report and machine-readable checks are:

- `results/oisa_real_validation_faa3c78/MFU_OISA_REAL_VALIDATION.md`
- `results/oisa_real_validation_faa3c78/validation.json`
- `results/oisa_real_validation_faa3c78/interface_a2a_comparison.json`
- `results/oisa_real_validation_faa3c78/iteration_comparison.csv`

Normal rebuilds use only the checked-in CSV/JSON files.  Refreshing trace facts
is optional and requires paths to the original authorized capture:

```bash
python scripts/import_pp_dag_trace.py \
  --source-case /path/to/source-case \
  --trace-root /path/to/framework_256_gbs_64 \
  --iteration 55

python scripts/import_pp_api_trace.py \
  --source-case /path/to/source-case \
  --trace-root /path/to/framework_256_gbs_64 \
  --iteration 55
```
