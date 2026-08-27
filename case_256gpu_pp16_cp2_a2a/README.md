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
- `results/oisa_topology_preflight_s5000_256_1spine`: reruns the same measured
  EDP-AG0 request with global ranks 230/238 on the 32-host S5000 topology.  It
  validates the corrected `gpus_per_server=8` header, 400-Gbps cross-host path,
  and natural runner exit.
- `results/oisa_s5000_256gpu_nine_class`: nine representative real-OISA
  requests/results for four EP A2A classes, CP A2A, DP RS/AG and Expert-DP
  RS/AG.  All preserve measured Rank release offsets and use one corrected
  topology hash.
- `results/pp_optimizer_dag_v4_oisa_s5000`: current model.  It separates raw
  OISA network tails from the signed Trace-minus-baseline-OISA residual, so a
  new topology changes only the network delta.  Same-hardware replay remains
  -0.1010% for ProfilerStep and +0.0954% relative for MFU.

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

The 256-GPU S5000 topology preflight evidence is:

- `results/oisa_topology_preflight_s5000_256_1spine/TOPOLOGY_PREFLIGHT.md`
- `results/oisa_topology_preflight_s5000_256_1spine/preflight_summary.json`

The current nine-class OISA/MFU outputs are:

- `results/oisa_s5000_256gpu_nine_class/oisa_results.csv`
- `results/oisa_s5000_256gpu_nine_class/OISA_NINE_CLASS_SUMMARY.md`
- `results/oisa_s5000_256gpu_nine_class/backfill/validation.json`
- `results/pp_optimizer_dag_v4_oisa_s5000/validation.json`
- `results/pp_optimizer_dag_v4_oisa_s5000/collective_slack_audit.csv`
- `results/pp_optimizer_dag_v4_oisa_s5000/PP_OPTIMIZER_DAG_V4_OISA_S5000.md`
- `results/pp_optimizer_dag_v4_oisa_s5000/pp_optimizer_dag_v4_oisa_s5000.html`

The final replay is calibrated, not a blind replacement by absolute ns-3 FCT.
For each class/request it keeps:

```text
target_tail = Trace_tail + (target_OISA_network - baseline_OISA_network)
```

Positive residuals can contain software/synchronization overhead; negative
residuals expose OISA algorithm, payload, or overlap bias and are retained as
calibration facts.  A raw network-only diagnostic is also emitted, but is not
the production MFU result.

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
