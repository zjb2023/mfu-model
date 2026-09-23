# 32GPU layer workload comparison r1

Scope: source32 PP1 L3–L6 / PP2 L7–L10, iterations [40, 45, 50, 55, 60, 65, 70, 75, 80], all 8 microbatches. The public comparison uses MB0–3 only, matching the 256GPU study. Forward work only; not a new timing prediction.

Chapter1 uses per-group counts: source32 single EP8 versus target256 (A+B)/2, for both retained assignments and active experts. Reference lines show pre-filter TOP6 assignments 8192*8*6=393216, or configured experts160 (an upper bound, not expected active count). Averaging may yield fractional expert counts. Same theoretical work does not establish equal actual workload or cost; average group work is not critical-path duration. Only chapter1 changes; other chapters retain their controls/data. Same iteration IDs are not matched inputs/checkpoints across runs. Original anchors remain valid.

18 representative traces (rank8 and rank16 across 9 iterations) locate four CPU CheckpointFunction layers per forward step. Same-rank timestamps anchor DeepEP calls. All eight ranks' sender counts sum exactly to the 160 receiver-expert counts, and representative FC split counts match the receiver log. 92160 per-expert conservation checks PASS. Nonrepresentative FC traces were not re-parsed. Raw inputs remain read-only; absolute paths, hashes, event indices and log line numbers are in the extraction files and manifest.

Shared-MB endpoint comparisons: L10<L3 in 35/36 cases; iter80>iter40 at 32/32 layer×MB positions. These are endpoint observations, not monotonicity or causal tests. Source depth cannot establish the long target depth trend.

Reproduce from the mfu-16to256 worktree:

```
/usr/bin/python3 -S workflow/data_foundation/extract_layer32_iteration_grid_r1.py
/usr/bin/python3 -S workflow/data_foundation/publish_layer32_comparison_r1.py
```

The extractor reuses verified cached per-stage JSON when present. Publish after existing layer-workload/pair/nine-iteration publishers if rebuilding the entire page. No prediction costs or frozen results were changed.
