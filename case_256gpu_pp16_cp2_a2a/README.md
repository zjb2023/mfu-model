# 256-GPU PP16 MFU case

The frozen case models iteration 55 from 256 ranks and 16 independent pipeline
lanes.  Each rank contributes four FWD and four BWD compute annotations.

## Versions

- `results/pp_dag_minimal`: FWD/BWD plus bidirectional PP network service.
- `results/pp_optimizer_dag_v1`: v0 frontier joined to DP/EDP RS and two AG rounds.
- `results/pp_optimizer_dag_v2`: backward network service and cluster software
  completion separated; current replay error is -0.1010% for ProfilerStep and
  +0.0954% relative for MFU.

The authoritative v2 explanation and validation are:

- `results/pp_optimizer_dag_v2/PP_OPTIMIZER_DAG_V2.md`
- `results/pp_optimizer_dag_v2/validation.json`
- `results/pp_optimizer_dag_v2/pp_software_completion_calibration.csv`

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
