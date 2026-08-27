# Trace-driven MFU model

This repository contains the standalone 256-GPU MFU replay developed for a
PP16 / CP2 / EP8 / DP8 training capture.  It builds a max-plus dependency graph
from per-microbatch FWD/BWD trace annotations, pipeline P2P messages, and
DP/Expert-DP RS/AG process-group facts.

The current `v2` model keeps network and software costs separate:

```text
backward PP completion
  = 4.712346 ms network service
  + 93.463844 ms cluster software completion
```

The software term is the median of 848 `send_backward` annotations from PP
stages 1–14.  It is not folded into the OISA/NS-3 network FCT.

## Current replay result

For captured iteration 55:

| Metric | Measured | Predicted | Error |
| --- | ---: | ---: | ---: |
| ProfilerStep | 21,638.095 ms | 21,616.236 ms | -0.1010% |
| Fixed-FLOP MFU | 4.4065% | 4.4107% | +0.0954% relative |

This is an in-sample replay result for one fully materialized iteration.  The
next validation step is to hold out additional captured iterations before
treating the software-completion parameter as a general predictor.

## Repository layout

- `src/x10000_analysis/pp_dag.py`: non-interleaved 1F1B schedule and PP DAG.
- `src/x10000_analysis/mfu_timeline.py`: DP/Expert-DP RS/AG max-plus model.
- `src/x10000_analysis/unified_mfu_dag.py`: dynamic PP-frontier to RS/AG join.
- `case_256gpu_pp16_cp2_a2a/data/`: frozen, compact model inputs; no raw traces.
- `case_256gpu_pp16_cp2_a2a/results/pp_dag_minimal/`: preserved PP-only v0.
- `case_256gpu_pp16_cp2_a2a/results/pp_optimizer_dag_v1/`: preserved merged v1.
- `case_256gpu_pp16_cp2_a2a/results/pp_optimizer_dag_v2/`: calibrated current model.

## Rebuild

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'

.venv/bin/python case_256gpu_pp16_cp2_a2a/scripts/build_pp_dag.py
.venv/bin/python case_256gpu_pp16_cp2_a2a/scripts/build_pp_optimizer_dag_v1.py
.venv/bin/python case_256gpu_pp16_cp2_a2a/scripts/build_pp_optimizer_dag_v2.py
.venv/bin/pytest -q
```

View the current self-contained dashboard:

```bash
python3 -m http.server 8013 --bind 127.0.0.1
```

Then open:

`http://127.0.0.1:8013/case_256gpu_pp16_cp2_a2a/results/pp_optimizer_dag_v2/pp_optimizer_dag_v2.html`

## Data boundary

The repository includes only derived timing facts needed to rebuild the model.
The original profiler JSON, training logs, topology dumps, and multi-gigabyte
kernel tables are intentionally excluded.  The optional `import_pp_*` scripts
require explicit `--source-case` and `--trace-root` paths when refreshing facts
on an authorized capture host.
