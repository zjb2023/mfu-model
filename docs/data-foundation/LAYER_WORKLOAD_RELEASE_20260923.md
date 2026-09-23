# Layer-workload analysis release — 2026-09-23

Branch: `32to256`. This release saves the nine-iteration 256GPU study and the appended 32GPU comparison. Frozen prediction parameters and raw trace/log data are unchanged. Only derived evidence, scripts, documentation and HTML are included.

## Results

- 256GPU: iterations 40/45/50/55/60/65/70/75/80, PP1–14, both EP8 groups, four microbatches. 645,120 expert conservation checks and 896 iter60 regression checks passed.
- 32GPU: the same nine iteration numbers, PP1–2, one EP8 group per stage, eight microbatches. 92,160 expert conservation checks passed. Public comparison uses shared MB0–3, comparing each 256GPU group separately with the 32GPU group.
- The latest browser check is `results/data-foundation/layer32-comparison-ui-r1/report.json`: 72 selector combinations, known-value assertions, existing heatmap, toggles, mobile layout and served sidecars passed.
- Source32 L10 has fewer retained assignments than L3 in 35/36 shared-MB endpoint comparisons. All 32 shared layer×MB positions have more retained assignments at iter80 than iter40. Neither result establishes monotonicity or causality.
- Identical iteration/layer labels across the two runs do not imply matched inputs, training state or weights. Counts are post-filter expert assignments, not unique tokens or durations.

## Rebuild

From the worktree root, after extraction/analysis artifacts are available:

```
/usr/bin/python3 -S workflow/data_foundation/publish_layer_workload_r1.py
/usr/bin/python3 -S workflow/data_foundation/publish_edp_pair_r1.py
/usr/bin/python3 -S workflow/data_foundation/publish_layer_iteration_grid_r1.py
/usr/bin/python3 -S workflow/data_foundation/publish_layer32_comparison_r1.py
```

The last publisher is idempotent; run it after the first three to preserve the appended comparison. Detailed extraction commands and limitations are in `LAYER_ITERATION_TRENDS_R1.md` and `LAYER32_WORKLOAD_COMPARISON_R1.md`.

Historical `layer-iteration-grid-ui-r1` and `layer-iteration-grid-verification-r1` record the previous section④-only HTML and extractor revision. Their hashes intentionally describe that earlier checkpoint, not the final appended HTML. The shared extractor was subsequently generalized with optional stage-width/microbatch/output settings for 32GPU; its default 256GPU logic was retained. The 32GPU manifest records the updated extractor hash. Do not interpret historical UI hashes as the current page hash.

No unrelated 16GPU cohort experiments are part of this release.
