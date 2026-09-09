"""Estimate FP8-peak-normalized model utilization from pinned MLPerf result_0 logs.
Reads cached inputs if present; otherwise fetches only three public result logs.
No training is executed. FLOPs convention: submission MODEL_TFLOP_PER_SAMPLE=1068.
"""
import json
import statistics
import urllib.request
from pathlib import Path

COMMIT = 'eabf23a07b2a0c60a289ff871dc3a46fff0d0421'
BASE = f'https://raw.githubusercontent.com/mlcommons/training_results_v6.0/{COMMIT}/'
CONFIG = 'NVIDIA/benchmarks/deepseekv3_671b/implementations/theia_ngpu512_ngc26.04_nemo/config_common.sh'
SPECS = ['https://www.nvidia.com/en-us/data-center/gb200-nvl72/', 'https://www.nvidia.com/en-us/data-center/gb300-nvl72/']
CASES = [('R12', 'GB300', 256, 'theia_ngpu256_ngc26.04_nemo'),
         ('R11', 'GB300', 512, 'theia_ngpu512_ngc26.04_nemo'),
         ('R13', 'GB200', 8192, 'tyche-hsg_ngpu8192_ngc26.04_nemo')]

def main():
    results = []
    for recipe, gpu, n, system in CASES:
        rel = f'NVIDIA/results/{system}/deepseekv3_671b/result_0.txt'
        cached = Path('/tmp/mlperf_recipe_research/sources/6.0') / rel
        raw = cached.read_text() if cached.exists() else urllib.request.urlopen(BASE + rel, timeout=60).read().decode()
        entries = [json.loads(line.split(':::MLLOG ', 1)[1]) for line in raw.splitlines() if ':::MLLOG ' in line]
        keys = {x['key'] for x in entries}
        # Two callbacks log train_step_time per step. Select only DeltaTimingCallback,
        # identified by reduced_train_loss. Its on_eval_end resets the timer.
        steps = [x['value']['train_step_time'] for x in entries
                 if x['key'] == 'tracked_stats' and isinstance(x['value'], dict)
                 and 'train_step_time' in x['value'] and 'reduced_train_loss' in x['value']]
        gbs = next(x['value'] for x in entries if x['key'] == 'global_batch_size')
        seq = next(x['value'] for x in entries if x['key'] == 'max_sequence_length')
        mean = statistics.mean(steps)
        achieved = 1068 * gbs / (n * mean)
        results.append(dict(recipe=recipe, gpu=gpu, gpus=n, gbs=gbs, seq_length=seq,
                            recorded_steps=len(steps), mean_step_seconds=mean,
                            median_step_seconds=statistics.median(steps),
                            model_tflop_per_sample=1068, peak_dense_fp8_tflops_per_gpu=5000,
                            achieved_model_tflops_per_gpu=achieved,
                            estimated_mfu_percent=100*achieved/5000,
                            direct_mfu_keys=[k for k in keys if 'mfu' in k.lower()],
                            result_url=f'https://github.com/mlcommons/training_results_v6.0/blob/{COMMIT}/{rel}'))
    output = dict(method='Single result_0; all DeltaTimingCallback train steps; ratio of model FLOPs to mean time and dense FP8 rated peak; not an official reported MFU.',
                  config_url=f'https://github.com/mlcommons/training_results_v6.0/blob/{COMMIT}/{CONFIG}',
                  peak_source_urls=SPECS, results=results)
    Path(__file__).with_suffix('.json').write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
    for x in results:
        print(x['recipe'],x['recorded_steps'],round(x['mean_step_seconds'],3),round(x['achieved_model_tflops_per_gpu'],1),round(x['estimated_mfu_percent'],2))

if __name__ == '__main__':
    main()
