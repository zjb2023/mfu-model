#!/usr/bin/env python3
"""Build a same-axis DAG v6.2/v6.3 prediction versus 224-GPU trace timeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO = Path(__file__).resolve().parents[2]
CASE = REPO / "case_224gpu_pp14_cp2_a2a"
DEFAULT_RUN = CASE / "results/mfu_accuracy_comparison_2026w36/dag_v62_cost_bound_256_to_224"
DEFAULT_MANIFEST = CASE / "results/readiness/profiler_trace_manifest.csv"
DEFAULT_TRACE_METADATA = CASE / "results/readiness/trace_metadata.csv"
DEFAULT_TRACE_ROOT = REPO / "fabric-data-analysis-raw/0731_224gpu_pp14_cp2_a2a/extracted/framework"
DEFAULT_COLLECTIVE_EVENTS = CASE / "results/collective_bw_no_pp/collective_events_no_pp_pre_mtlink.parquet"
SOURCE_256_PHASE_EVENTS = REPO / (
    "case_256gpu_pp16_cp2_a2a/results/mfu_accuracy_comparison_2026w36/"
    "dag_v53_ep_group_source60_100/source_pp_trace_events_60_100.csv"
)
SOURCE_256_COLLECTIVE_EVENTS = REPO / (
    "case_256gpu_pp16_cp2_a2a/results/mfu_accuracy_comparison_2026w36/"
    "dag_v54_source60_100_bundle/source_stable_collective_rank_events.csv"
)
ITERATIONS = tuple(range(60, 101, 5))
PHASE_COLORS = {"forward": "#24c8ff", "backward": "#ff725c"}
PHASE_TRACKS = {
    "forward": {"y_offset": 3, "height": 9},
    "backward": {"y_offset": 17, "height": 9},
}
HTML_EVOLUTION = (
    ("v6.2", "建立 PP14×3 的预测/Trace 同轴总览，并显示 F/B 依赖箭头。"),
    ("v6.3", "补齐 DP RS→Expert-DP RS，以及 DP AG→Expert-DP AG 的程序顺序。"),
    ("v6.3.1", "把 AG0→AG1 的 stage 同步从隐式等待改成可见节点。"),
    ("v6.4a", "加入 F→B、B→F 调度交接节点，保留 1F1B 局部依赖。"),
    ("v6.5", "按真实 52 层 [2, 4×12, 2] 转移全局 layer 模板。"),
    ("v6.6", "拆分 optimizer 尾部并完成 256 卡回放；预测/Trace 共用 F/B 颜色与上下轨布局。"),
    ("v6.6-ui1", "将预测阶段由 fwd/bwd 标准化为 forward/backward，修复 A 图全部落入 BWD 下轨的问题。"),
    ("v6.7", "从256卡全部rank学习stage/lane/microbatch相对时长形状，并显示源p10–p90运行波动带。"),
    ("v6.7-ui1", "去除F/B外侧的p10–p90虚线框；波动数据仍保留在结果文件中，不再叠加到主时序图。"),
    ("v6.7-ui2", "同页加入256卡源Trace（PP16×4），并隐藏预测图中仅用于调度建模的B→F紫色handoff条。"),
    ("v6.7-ui3", "A图不再绘制任何F/B调度handoff条，去除被误标为DAG group service的紫色长条；模型依赖保持不变。"),
    ("v6.7-ui4", "将B/C的DP RS rank kernel拆成早到驻留与最后rank到达后尾部，并增加224/256同口径汇总。"),
    ("v6.7-ui5", "B/C的DP RS统一改用全组Trace包络：最早rank到达、最晚rank到达、全组完成；lane2仅保留为tooltip诊断。"),
    ("v6.7-ui6", "A图补画DAG已有的16-rank到达跨度，并去掉“释放门槛”的旧叫法；只改变展示，不重复增加时长。"),
    ("v6.7-ui7", "A/B/C的F/B统一改成每个PP stage、每个microbatch覆盖16个rank的时间包络，并将该等待量直白表述为“相邻F/B启动间隔”。"),
)
TAIL_BEHAVIORS = {
    "dp_grad_reduce_scatter": ("dp", "rs", 0),
    "expert_dp_grad_reduce_scatter": ("expert_dp", "rs", 0),
    "dp_param_allgather": ("dp", "ag", None),
    "expert_dp_param_allgather": ("expert_dp", "ag", None),
}
ANNOTATION = re.compile(
    rb'"ph": "X", "cat": "user_annotation", "name": "([^"]+)"[^\n]*'
    rb'\n\s*"ts": ([0-9.]+), "dur": ([0-9.]+),'
)
RG_ANNOTATION_PATTERN = (
    r'"ph": "X", "cat": "user_annotation", "name": '
    r'"(?:forward_step|backward_step)"[^\n]*\n\s*'
    r'"ts": [0-9.]+, "dur": [0-9.]+'
)


def display_version(model_version: str) -> str:
    return {
        "v62": "v6.2", "v63": "v6.3", "v631": "v6.3.1",
        "v64a": "v6.4a", "v65": "v6.5", "v66": "v6.6", "v67": "v6.7",
    }[model_version]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked(path: Path) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        frame.to_csv(handle, index=False)
        temporary = Path(handle.name)
    temporary.replace(path)


def expected_schedule(stage: int, pp: int = 14, microbatches: int = 3) -> list[str]:
    warmup = min(pp - stage - 1, microbatches)
    remaining = microbatches - warmup
    operations: list[tuple[str, int]] = [("forward", mb) for mb in range(warmup)]
    for index in range(remaining):
        operations.append(("forward", warmup + index))
        operations.append(("backward", index))
    operations.extend(("backward", mb) for mb in range(remaining, microbatches))
    return [f"{'F' if phase == 'forward' else 'B'}{mb}" for phase, mb in operations]


def semantic_dependencies(pp: int = 14, microbatches: int = 3) -> list[dict[str, Any]]:
    dependencies: list[dict[str, Any]] = []
    for stage in range(pp):
        sequence = expected_schedule(stage, pp, microbatches)
        for source, target in zip(sequence, sequence[1:]):
            dependencies.append({
                "source": f"s{stage}:{source}", "target": f"s{stage}:{target}",
                "kind": "rank_program_order", "stage": stage,
                "label": f"PP{stage} local 1F1B: {source} → {target}",
            })
        for microbatch in range(microbatches):
            dependencies.append({
                "source": f"s{stage}:F{microbatch}", "target": f"s{stage}:B{microbatch}",
                "kind": "autograd_activation", "stage": stage,
                "label": f"PP{stage} autograd activation: F{microbatch} → B{microbatch}",
            })
    for stage in range(pp - 1):
        for microbatch in range(microbatches):
            dependencies.append({
                "source": f"s{stage}:F{microbatch}", "target": f"s{stage + 1}:F{microbatch}",
                "kind": "pp_activation", "stage": stage,
                "label": f"activation F{microbatch}: PP{stage} → PP{stage + 1}",
            })
            dependencies.append({
                "source": f"s{stage + 1}:B{microbatch}", "target": f"s{stage}:B{microbatch}",
                "kind": "pp_gradient", "stage": stage,
                "label": f"gradient B{microbatch}: PP{stage + 1} → PP{stage}",
            })
    return dependencies


def phase_rows_from_raw(
    raw: list[tuple[float, str, float]],
    path: Path,
    iteration: int,
    rank: int,
    stage: int,
    lane: int,
) -> list[dict[str, Any]]:
    occurrences: Counter[str] = Counter()
    rows = []
    symbols = []
    for start_us, name, duration_us in sorted(raw):
        phase = "forward" if name == "forward_step" else "backward"
        microbatch = occurrences[phase]
        occurrences[phase] += 1
        symbols.append(f"{'F' if phase == 'forward' else 'B'}{microbatch}")
        rows.append({
            "iteration": iteration, "rank": rank, "pp_stage": stage, "pp_lane": lane,
            "phase": phase, "microbatch": microbatch,
            "timestamp_ns": int(round(start_us * 1000.0)),
            "duration_ns": int(round(duration_us * 1000.0)),
            "source_path": str(path.resolve()),
        })
    expected = expected_schedule(stage)
    if symbols != expected:
        raise ValueError(f"trace schedule mismatch rank={rank} iteration={iteration}: {symbols} != {expected}")
    return rows


def parse_trace(path: Path, iteration: int, rank: int, stage: int, lane: int) -> list[dict[str, Any]]:
    raw = []
    for match in ANNOTATION.finditer(checked(path).read_bytes()):
        name = match.group(1)
        if name not in {b"forward_step", b"backward_step"}:
            continue
        raw.append((float(match.group(2)), name.decode(), float(match.group(3))))
    return phase_rows_from_raw(raw, path, iteration, rank, stage, lane)


def interval_union_ns(frame: pd.DataFrame, start_col: str, end_col: str) -> int:
    intervals = sorted(
        (int(start), int(end))
        for start, end in zip(frame[start_col], frame[end_col])
    )
    if not intervals:
        return 0
    union = 0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            union += current_end - current_start
            current_start, current_end = start, end
    return union + current_end - current_start


def aggregate_phase_envelopes(
    frame: pd.DataFrame,
    *,
    start_col: str,
    end_col: str,
    stage_count: int,
    microbatch_count: int,
    expected_rank_count: int = 16,
    include_iteration: bool = True,
) -> pd.DataFrame:
    keys = (["iteration"] if include_iteration else []) + ["pp_stage", "phase", "microbatch"]
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(keys, sort=True):
        values = key if isinstance(key, tuple) else (key,)
        record = dict(zip(keys, values))
        record.update({
            "first_start_ns": int(group[start_col].min()),
            "last_start_ns": int(group[start_col].max()),
            "end_ns": int(group[end_col].max()),
            "rank_count": int(group["rank"].nunique()),
            "active_union_ns": interval_union_ns(group, start_col, end_col),
            "min_rank_duration_ns": int((group[end_col] - group[start_col]).min()),
            "median_rank_duration_ns": int((group[end_col] - group[start_col]).median()),
            "max_rank_duration_ns": int((group[end_col] - group[start_col]).max()),
        })
        record["duration_ns"] = record["end_ns"] - record["first_start_ns"]
        rows.append(record)
    result = pd.DataFrame(rows)
    expected_rows = stage_count * 2 * microbatch_count
    if include_iteration:
        expected_rows *= int(frame["iteration"].nunique())
    if len(result) != expected_rows:
        raise ValueError(f"phase envelope grid incomplete: {len(result)} != {expected_rows}")
    if not result["rank_count"].eq(expected_rank_count).all():
        bad = result.loc[~result["rank_count"].eq(expected_rank_count), keys + ["rank_count"]]
        raise ValueError(f"phase envelope rank grid incomplete: {bad.head().to_dict('records')}")
    return result


def extract_target_phase_iteration(
    selected: pd.DataFrame,
    trace_root: Path,
    iteration: int,
) -> pd.DataFrame:
    iteration_rows = selected[selected["iter"].eq(iteration)].sort_values("rank")
    if len(iteration_rows) != 224 or iteration_rows["rank"].nunique() != 224:
        raise ValueError(f"target224 raw phase trace grid incomplete at iteration {iteration}")
    metadata: dict[str, tuple[int, int, int, Path]] = {}
    paths: list[str] = []
    for row in iteration_rows.itertuples(index=False):
        path = checked(trace_root / str(row.relative_path))
        metadata[str(path)] = (int(row.rank), int(row.rank) // 16, int(row.rank) % 16, path)
        paths.append(str(path))
    command = ["rg", "-a", "--json", "-U", "-o", "--pcre2", RG_ANNOTATION_PATTERN, *paths]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    raw_by_path: dict[str, list[tuple[float, str, float]]] = defaultdict(list)
    for line in completed.stdout.splitlines():
        message = json.loads(line)
        if message.get("type") != "match":
            continue
        path = str(Path(message["data"]["path"]["text"]).resolve())
        match = ANNOTATION.search(message["data"]["lines"]["text"].encode())
        if match is None:
            raise ValueError(f"cannot decode rg phase match: {path}")
        raw_by_path[path].append(
            (float(match.group(2)), match.group(1).decode(), float(match.group(3)))
        )
    rows: list[dict[str, Any]] = []
    for path_text, (rank, stage, lane, path) in metadata.items():
        rows.extend(
            phase_rows_from_raw(raw_by_path.get(path_text, []), path, iteration, rank, stage, lane)
        )
    events = pd.DataFrame(rows)
    events["end_ns"] = events["timestamp_ns"] + events["duration_ns"]
    return aggregate_phase_envelopes(
        events,
        start_col="timestamp_ns",
        end_col="end_ns",
        stage_count=14,
        microbatch_count=3,
    )


def load_target_phase_envelopes(
    manifest: pd.DataFrame,
    trace_root: Path,
    evaluator: Path,
) -> tuple[pd.DataFrame, Path, list[Path]]:
    selected = manifest[manifest["iter"].isin(ITERATIONS)].copy()
    if len(selected) != len(ITERATIONS) * 224:
        raise ValueError(f"target224 trace manifest incomplete: {len(selected)}")
    selected["absolute_path"] = selected["relative_path"].map(
        lambda value: str(checked(trace_root / str(value)))
    )
    selection_manifest = evaluator / "timeline_trace_stage16_phase_input_manifest_60_100.csv"
    atomic_csv(selection_manifest, selected.sort_values(["iter", "rank"]))
    caches: list[Path] = []
    frames: list[pd.DataFrame] = []
    required = {
        "iteration", "pp_stage", "phase", "microbatch", "first_start_ns",
        "last_start_ns", "end_ns", "rank_count", "active_union_ns",
        "median_rank_duration_ns", "duration_ns",
    }
    for iteration in ITERATIONS:
        cache = evaluator / f"timeline_trace_stage16_phase_envelope_iter{iteration}.csv"
        caches.append(cache)
        frame = pd.read_csv(cache) if cache.is_file() else None
        if frame is None or not required.issubset(frame.columns):
            print(f"[phase-envelope] extracting target224 iteration {iteration}", flush=True)
            frame = extract_target_phase_iteration(selected, trace_root, iteration)
            atomic_csv(cache, frame)
        if len(frame) != 14 * 2 * 3 or not required.issubset(frame.columns):
            raise ValueError(f"cached target224 phase envelope changed: {cache}")
        if not frame["rank_count"].eq(16).all():
            raise ValueError(f"cached target224 phase envelope is not 16-rank complete: {cache}")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), selection_manifest, caches


def phase_decomposition(profiler_ms: float, start_ms: pd.Series, end_ms: pd.Series) -> dict[str, float]:
    phase_span = float(end_ms.max() - start_ms.min())
    outside = float(profiler_ms - phase_span)
    if outside < -1e-6:
        raise ValueError(f"phase envelope exceeds profiler clock: {phase_span} > {profiler_ms}")
    return {
        "profiler_ms": float(profiler_ms),
        "phase_envelope_ms": phase_span,
        "outside_phase_envelope_ms": max(outside, 0.0),
    }


def adjacent_phase_gaps(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage in range(14):
        row: dict[str, Any] = {"stage": stage}
        for phase, prefix in (("forward", "f"), ("backward", "b")):
            selected = frame[frame["stage"].eq(stage) & frame["phase"].eq(phase)].set_index("microbatch")
            if set(selected.index.astype(int)) != {0, 1, 2}:
                raise ValueError(f"adjacent-gap phase grid incomplete: stage={stage}, phase={phase}")
            row[f"{prefix}01_ms"] = round(float(selected.loc[1, "start_ms"] - selected.loc[0, "end_ms"]), 6)
            row[f"{prefix}12_ms"] = round(float(selected.loc[2, "start_ms"] - selected.loc[1, "end_ms"]), 6)
        rows.append(row)
    return rows


def predicted_collective_bars(nodes: pd.DataFrame, lane: int, origin_ns: int) -> list[dict[str, Any]]:
    # A PP stage has 16 lanes because CP=2 and DP=8, while the Expert-DP tail
    # has eight groups per stage.  PP lanes 0/8, 1/9, ... therefore share the
    # same Expert-DP group node.
    expert_dp_lane = lane % 8
    services = nodes[nodes["kind"].eq("collective_service")].copy()
    services = services[
        services["node_id"].str.match(r"^tail:dp_with_cp_stage[0-9]+:(rs|ag0|ag1)_service$")
        | services["node_id"].str.match(
            rf"^tail:expert_dp_stage[0-9]+_lane{expert_dp_lane}:(rs|ag0|ag1)_service$"
        )
    ].copy()
    if len(services) != 14 * 6:
        raise ValueError(f"predicted DP/EDP grid incomplete for lane{lane}: {len(services)}")
    dp_rs_release = nodes[
        nodes["kind"].eq("collective_release_token")
        & nodes["node_id"].str.match(
            r"^tail:dp_with_cp_stage[0-9]+:rs_arrival_join::release:r[0-9]+$"
        )
    ].copy()
    dp_rs_release_by_stage: dict[int, dict[str, Any]] = {}
    if not dp_rs_release.empty:
        dp_rs_release_groups = dp_rs_release.groupby("pp_stage", as_index=False).agg(
            group_observed_rank_count=("rank", "nunique"),
            first_release_ns=("predicted_end_ns", "min"),
            last_release_ns=("predicted_end_ns", "max"),
        )
        if (
            len(dp_rs_release_groups) != 14
            or not dp_rs_release_groups["group_observed_rank_count"].eq(16).all()
        ):
            raise ValueError("predicted DP RS release grid incomplete")
        dp_rs_release_by_stage = dp_rs_release_groups.set_index("pp_stage").to_dict("index")
    rows = []
    for row in services.sort_values(["pp_stage", "predicted_start_ns"]).itertuples(index=False):
        node_id = str(row.node_id)
        collective = node_id.rsplit(":", 1)[1].removesuffix("_service")
        group_type = "expert_dp" if ":expert_dp_" in node_id else "dp"
        bar = {
            "id": node_id,
            "stage": int(row.pp_stage),
            "group_type": group_type,
            "collective": collective,
            "start_ms": round((int(row.predicted_start_ns) - origin_ns) / 1e6, 6),
            "end_ms": round((int(row.predicted_end_ns) - origin_ns) / 1e6, 6),
            "duration_ms": round(int(row.duration_ns) / 1e6, 6),
            "network_ms": round(int(row.network_service_ns_model) / 1e6, 6),
            "software_sync_ms": round(int(row.software_sync_ns_model) / 1e6, 6),
            "critical": bool(row.on_critical_path),
        }
        if (
            group_type == "dp"
            and collective == "rs"
            and int(row.pp_stage) in dp_rs_release_by_stage
        ):
            release = dp_rs_release_by_stage[int(row.pp_stage)]
            first_release_ns = int(release["first_release_ns"])
            last_release_ns = int(release["last_release_ns"])
            service_start_ns = int(row.predicted_start_ns)
            service_end_ns = int(row.predicted_end_ns)
            if last_release_ns != service_start_ns:
                raise ValueError(
                    f"predicted DP RS join changed at stage {int(row.pp_stage)}: "
                    f"last release {last_release_ns} != service start {service_start_ns}"
                )
            bar.update({
                "service_start_ms": bar["start_ms"],
                "service_duration_ms": bar["duration_ms"],
                "start_ms": round((first_release_ns - origin_ns) / 1e6, 6),
                "last_release_ms": round((last_release_ns - origin_ns) / 1e6, 6),
                "end_ms": round((service_end_ns - origin_ns) / 1e6, 6),
                "duration_ms": round((service_end_ns - first_release_ns) / 1e6, 6),
                "group_arrival_span_ms": round(
                    (last_release_ns - first_release_ns) / 1e6, 6
                ),
                "group_tail_after_last_release_ms": round(
                    (service_end_ns - last_release_ns) / 1e6, 6
                ),
                "group_observed_rank_count": int(release["group_observed_rank_count"]),
                "group_expected_rank_count": 16,
            })
        rows.append(bar)
    barriers = nodes[nodes["node_id"].str.match(r"^tail:stage[0-9]+:ag0_round_barrier$")].copy()
    if not barriers.empty:
        if len(barriers) != 14:
            raise ValueError(f"predicted AG round barrier grid incomplete: {len(barriers)}")
        for row in barriers.sort_values("pp_stage").itertuples(index=False):
            rows.append({
                "id": str(row.node_id), "stage": int(row.pp_stage),
                "group_type": "barrier", "collective": "ag_round",
                "start_ms": round((int(row.predicted_start_ns) - origin_ns) / 1e6, 6),
                "end_ms": round((int(row.predicted_end_ns) - origin_ns) / 1e6, 6),
                "duration_ms": round(int(row.duration_ns) / 1e6, 6),
                "network_ms": 0.0,
                "software_sync_ms": round(int(row.software_sync_ns_model) / 1e6, 6),
                "critical": bool(row.on_critical_path),
            })
    handoffs = nodes[
        nodes["kind"].eq("scheduler_handoff") & nodes["pp_lane"].eq(lane)
    ].copy()
    if not handoffs.empty:
        if len(handoffs) != 20:
            raise ValueError(f"predicted phase handoff grid incomplete for lane{lane}: {len(handoffs)}")
        for row in handoffs.sort_values(["pp_stage", "predicted_start_ns"]).itertuples(index=False):
            rows.append({
                "id": str(row.node_id), "stage": int(row.pp_stage),
                "group_type": "handoff", "collective": str(row.semantic_slot).lower(),
                "start_ms": round((int(row.predicted_start_ns) - origin_ns) / 1e6, 6),
                "end_ms": round((int(row.predicted_end_ns) - origin_ns) / 1e6, 6),
                "duration_ms": round(int(row.duration_ns) / 1e6, 6),
                "network_ms": 0.0,
                "software_sync_ms": round(int(row.software_sync_ns_model) / 1e6, 6),
                "critical": bool(row.on_critical_path),
            })
    optimizer_columns = {"rank", "op_name"}
    optimizer = (
        nodes[
            nodes["rank"].ge(0)
            & nodes["rank"].mod(16).eq(lane)
            & nodes["op_name"].isin(["optimizer_update", "optimizer_pre_ag0", "optimizer_post_ag0"])
        ].copy()
        if optimizer_columns.issubset(nodes.columns)
        else nodes.iloc[0:0].copy()
    )
    if not optimizer.empty:
        if len(optimizer) != 14 * 3:
            raise ValueError(f"predicted optimizer stream grid incomplete for lane{lane}: {len(optimizer)}")
        for row in optimizer.sort_values(["pp_stage", "predicted_start_ns"]).itertuples(index=False):
            rows.append({
                "id": str(row.node_id), "stage": int(row.pp_stage),
                "group_type": "optimizer", "collective": str(row.op_name),
                "start_ms": round((int(row.predicted_start_ns) - origin_ns) / 1e6, 6),
                "end_ms": round((int(row.predicted_end_ns) - origin_ns) / 1e6, 6),
                "duration_ms": round(int(row.duration_ns) / 1e6, 6),
                "network_ms": 0.0,
                "software_sync_ms": 0.0,
                "critical": bool(row.on_critical_path),
            })
    return rows


def load_trace_collectives(
    parquet_path: Path,
    trace_metadata_path: Path,
    cache: Path,
    lane: int,
) -> pd.DataFrame:
    expected_rows = len(ITERATIONS) * 14 * 6
    split_columns = {
        "dp_group_first_release_trace_ns", "dp_group_last_release_trace_ns",
        "dp_group_end_trace_ns", "dp_group_arrival_span_ns",
        "dp_group_tail_after_last_release_ns", "dp_group_key",
        "dp_group_observed_rank_count", "dp_group_expected_rank_count",
    }
    if cache.is_file():
        frame = pd.read_csv(cache)
    if (
        not cache.is_file()
        or not {"trace_start_ns", "trace_end_ns", *split_columns}.issubset(frame.columns)
    ):
        columns = [
            "event_id", "behavior", "iteration", "rank", "start_ns", "end_ns",
            "duration_ns", "logical_input_bytes", "source_path", "pg_name",
            "group_size",
        ]
        try:
            full = pd.read_parquet(checked(parquet_path), columns=columns)
        except ImportError as error:
            raise RuntimeError(
                "reading target collective trace requires the project .venv: "
                ".venv/bin/python case_224gpu_pp14_cp2_a2a/scripts/"
                "build_dag_v62_trace_timeline_comparison.py"
            ) from error
        dp_rs = full[
            full["iteration"].isin(ITERATIONS)
            & full["behavior"].eq("dp_grad_reduce_scatter")
        ].copy()
        dp_rs["pp_stage"] = dp_rs["rank"].floordiv(16).astype(int)
        dp_rs["dp_group_key"] = dp_rs["pg_name"].astype(str)
        dp_key = ["iteration", "pp_stage", "dp_group_key"]
        dp_groups = dp_rs.groupby(dp_key, as_index=False).agg(
            dp_group_observed_rank_count=("rank", "nunique"),
            dp_group_expected_rank_count=("group_size", "first"),
            dp_group_first_release_ns=("start_ns", "min"),
            dp_group_last_release_ns=("start_ns", "max"),
            dp_group_end_ns=("end_ns", "max"),
        )
        if not (
            dp_groups["dp_group_observed_rank_count"]
            .eq(dp_groups["dp_group_expected_rank_count"])
            .all()
            and dp_groups["dp_group_expected_rank_count"].eq(16).all()
        ):
            raise ValueError("target224 DP RS group is incomplete")
        dp_groups["dp_group_arrival_span_ns"] = (
            dp_groups["dp_group_last_release_ns"] - dp_groups["dp_group_first_release_ns"]
        )
        dp_groups["dp_group_tail_after_last_release_ns"] = (
            dp_groups["dp_group_end_ns"] - dp_groups["dp_group_last_release_ns"]
        )
        ranks = {stage * 16 + lane for stage in range(14)}
        frame = full[
            full["iteration"].isin(ITERATIONS)
            & full["rank"].isin(ranks)
            & full["behavior"].isin(TAIL_BEHAVIORS)
        ].copy()
        del full
        metadata = pd.read_csv(
            checked(trace_metadata_path), usecols=["relative_path", "base_time_ns"]
        )
        frame = frame.merge(
            metadata,
            left_on="source_path",
            right_on="relative_path",
            how="left",
            validate="many_to_one",
        )
        if frame["base_time_ns"].isna().any():
            raise ValueError("trace collective clock metadata is incomplete")
        frame["trace_start_ns"] = frame["start_ns"] - frame["base_time_ns"]
        frame["trace_end_ns"] = frame["end_ns"] - frame["base_time_ns"]
        frame["pp_stage"] = frame["rank"].floordiv(16).astype(int)
        frame["dp_group_key"] = frame["pg_name"].astype(str)
        frame = frame.merge(
            dp_groups,
            on=dp_key,
            how="left",
            validate="many_to_one",
        )
        frame["dp_group_last_release_trace_ns"] = (
            frame["dp_group_last_release_ns"] - frame["base_time_ns"]
        )
        frame["dp_group_first_release_trace_ns"] = (
            frame["dp_group_first_release_ns"] - frame["base_time_ns"]
        )
        frame["dp_group_end_trace_ns"] = (
            frame["dp_group_end_ns"] - frame["base_time_ns"]
        )
        frame["group_type"] = frame["behavior"].map(
            lambda value: TAIL_BEHAVIORS[str(value)][0]
        )
        frame["collective"] = frame["behavior"].map(
            lambda value: TAIL_BEHAVIORS[str(value)][1]
        )
        frame["round"] = frame.groupby(
            ["iteration", "rank", "behavior"], sort=False
        ).cumcount()
        frame.loc[frame["collective"].eq("ag"), "collective"] += frame.loc[
            frame["collective"].eq("ag"), "round"
        ].astype(str)
        frame = frame.sort_values(["iteration", "rank", "start_ns"]).reset_index(drop=True)
        atomic_csv(cache, frame)
    if len(frame) != expected_rows:
        raise ValueError(f"trace DP/EDP grid incomplete for lane{lane}: {len(frame)} != {expected_rows}")
    expected = {"dp:rs", "dp:ag0", "dp:ag1", "expert_dp:rs", "expert_dp:ag0", "expert_dp:ag1"}
    observed = set(frame["group_type"].astype(str) + ":" + frame["collective"].astype(str))
    if observed != expected:
        raise ValueError(f"trace DP/EDP semantics changed: {observed}")
    return frame


def build_source_256_trace_payload(
    phase_events_path: Path,
    collective_events_path: Path,
    lane: int,
) -> dict[str, Any]:
    """Build a display-only PP16 x M4 source trace on the same relative clock."""
    phases = pd.read_csv(checked(phase_events_path))
    required_phase_columns = {
        "iteration", "rank", "pp_stage", "pp_lane", "phase", "microbatch",
        "observed_start_ns", "observed_end_ns", "duration_ns",
    }
    if not required_phase_columns.issubset(phases.columns):
        raise ValueError("source256 phase trace schema changed")
    phases = phases[phases["iteration"].isin(ITERATIONS)].copy()
    expected_phase_rows = len(ITERATIONS) * 256 * 2 * 4
    if len(phases) != expected_phase_rows:
        raise ValueError(
            "source256 PP16x4 phase grid incomplete across all ranks: "
            f"{len(phases)} != {expected_phase_rows}"
        )
    for (iteration, rank), rank_events in phases.groupby(["iteration", "rank"]):
        ordered = rank_events.sort_values("observed_start_ns")
        symbols = [
            f"{'F' if str(row.phase) == 'forward' else 'B'}{int(row.microbatch)}"
            for row in ordered.itertuples(index=False)
        ]
        expected = expected_schedule(int(rank) // 16, pp=16, microbatches=4)
        if symbols != expected:
            raise ValueError(
                f"source256 trace schedule mismatch rank={rank} iteration={iteration}: "
                f"{symbols} != {expected}"
            )
    phase_envelopes = aggregate_phase_envelopes(
        phases,
        start_col="observed_start_ns",
        end_col="observed_end_ns",
        stage_count=16,
        microbatch_count=4,
    )

    all_collectives = pd.read_csv(checked(collective_events_path))
    required_collective_columns = {
        "iteration", "behavior", "group_type", "group_id", "call_round",
        "rank", "pp_stage", "pp_lane", "group_size", "start_ns", "end_ns",
    }
    if not required_collective_columns.issubset(all_collectives.columns):
        raise ValueError("source256 collective rank-event schema changed")
    all_collectives = all_collectives[
        all_collectives["iteration"].isin(ITERATIONS)
    ].copy()
    dp_rs = all_collectives[
        all_collectives["behavior"].eq("dp_grad_reduce_scatter")
    ].copy()
    dp_key = ["iteration", "behavior", "group_type", "group_id", "call_round"]
    dp_groups = dp_rs.groupby(dp_key, as_index=False).agg(
        dp_group_observed_rank_count=("rank", "nunique"),
        dp_group_expected_rank_count=("group_size", "first"),
        dp_group_first_release_ns=("start_ns", "min"),
        dp_group_last_release_ns=("start_ns", "max"),
        dp_group_end_ns=("end_ns", "max"),
    )
    if not (
        dp_groups["dp_group_observed_rank_count"]
        .eq(dp_groups["dp_group_expected_rank_count"])
        .all()
        and dp_groups["dp_group_expected_rank_count"].eq(16).all()
    ):
        raise ValueError("source256 DP RS group is incomplete")
    dp_groups["dp_group_arrival_span_ns"] = (
        dp_groups["dp_group_last_release_ns"] - dp_groups["dp_group_first_release_ns"]
    )
    dp_groups["dp_group_tail_after_last_release_ns"] = (
        dp_groups["dp_group_end_ns"] - dp_groups["dp_group_last_release_ns"]
    )
    collectives = all_collectives[all_collectives["pp_lane"].eq(lane)].copy()
    collectives = collectives.merge(
        dp_groups,
        on=dp_key,
        how="left",
        validate="many_to_one",
    )
    behavior_to_collective = {
        "dp_grad_reduce_scatter": "rs",
        "expert_dp_grad_reduce_scatter": "rs",
        "dp_param_allgather": "ag",
        "expert_dp_param_allgather": "ag",
    }
    unknown = set(collectives["behavior"].astype(str)) - set(behavior_to_collective)
    if unknown:
        raise ValueError(f"source256 collective behavior changed: {sorted(unknown)}")

    payload: dict[str, Any] = {}
    expected_collectives_per_iteration = 16 * 6
    for iteration in ITERATIONS:
        frame = phase_envelopes[phase_envelopes["iteration"].eq(iteration)].copy()
        frame = frame.sort_values(["first_start_ns", "pp_stage", "microbatch"])
        if len(frame) != 16 * 2 * 4:
            raise ValueError(f"source256 phase iteration {iteration} is incomplete")
        origin = int(frame["first_start_ns"].min())
        bars = [
            {
                "id": f"s{int(row.pp_stage)}:{'F' if str(row.phase) == 'forward' else 'B'}{int(row.microbatch)}",
                "stage": int(row.pp_stage),
                "phase": str(row.phase),
                "microbatch": int(row.microbatch),
                "start_ms": round((int(row.first_start_ns) - origin) / 1e6, 6),
                "last_start_ms": round((int(row.last_start_ns) - origin) / 1e6, 6),
                "end_ms": round((int(row.end_ns) - origin) / 1e6, 6),
                "duration_ms": round(int(row.duration_ns) / 1e6, 6),
                "active_union_ms": round(int(row.active_union_ns) / 1e6, 6),
                "rank_count": int(row.rank_count),
            }
            for row in frame.itertuples(index=False)
        ]

        collective_frame = collectives[collectives["iteration"].eq(iteration)].copy()
        collective_frame = collective_frame.sort_values(["start_ns", "pp_stage", "behavior"])
        collective_bars = []
        for row in collective_frame.itertuples(index=False):
            behavior = str(row.behavior)
            collective = behavior_to_collective[behavior]
            if collective == "ag":
                collective = f"ag{int(row.call_round)}"
            group_type = "dp" if str(row.group_type) == "dp_with_cp" else "expert_dp"
            bar = {
                "id": (
                    f"source256:{iteration}:rank{int(row.rank)}:{behavior}:"
                    f"round{int(row.call_round)}"
                ),
                "stage": int(row.pp_stage),
                "rank": int(row.rank),
                "group_type": group_type,
                "collective": collective,
                "start_ms": round((int(row.start_ns) - origin) / 1e6, 6),
                "end_ms": round((int(row.end_ns) - origin) / 1e6, 6),
                "duration_ms": round((int(row.end_ns) - int(row.start_ns)) / 1e6, 6),
            }
            if behavior == "dp_grad_reduce_scatter":
                first_release_ns = int(row.dp_group_first_release_ns)
                last_release_ns = int(row.dp_group_last_release_ns)
                group_end_ns = int(row.dp_group_end_ns)
                bar.update({
                    "rank_start_ms": bar["start_ms"],
                    "rank_end_ms": bar["end_ms"],
                    "rank_duration_ms": bar["duration_ms"],
                    "group_key": str(row.group_id),
                    "group_observed_rank_count": int(row.dp_group_observed_rank_count),
                    "group_expected_rank_count": int(row.dp_group_expected_rank_count),
                    "start_ms": round((first_release_ns - origin) / 1e6, 6),
                    "last_release_ms": round((last_release_ns - origin) / 1e6, 6),
                    "end_ms": round((group_end_ns - origin) / 1e6, 6),
                    "duration_ms": round((group_end_ns - first_release_ns) / 1e6, 6),
                    "early_residence_ms": round(
                        max(0, min(int(row.end_ns), last_release_ns) - int(row.start_ns)) / 1e6,
                        6,
                    ),
                    "rank_post_last_release_ms": round(
                        max(0, int(row.end_ns) - max(int(row.start_ns), last_release_ns)) / 1e6,
                        6,
                    ),
                    "group_arrival_span_ms": round(int(row.dp_group_arrival_span_ns) / 1e6, 6),
                    "group_tail_after_last_release_ms": round(
                        int(row.dp_group_tail_after_last_release_ns) / 1e6, 6
                    ),
                })
            collective_bars.append(bar)
        observed_collectives = len(collective_bars)
        if observed_collectives > expected_collectives_per_iteration:
            raise ValueError(
                f"source256 collective iteration {iteration} has unexpected duplicates: "
                f"{observed_collectives} > {expected_collectives_per_iteration}"
            )
        timeline_end_ms = max(
            max(item["end_ms"] for item in bars),
            max(item["end_ms"] for item in collective_bars),
        )
        payload[str(iteration)] = {
            "bars": bars,
            "collectives": collective_bars,
            "stage_count": 16,
            "microbatch_count": 4,
            "timeline_end_ms": round(timeline_end_ms, 6),
            "phase_envelope_ms": round(
                max(item["end_ms"] for item in bars) - min(item["start_ms"] for item in bars), 6
            ),
            "observed_collective_count": observed_collectives,
            "expected_collective_count": expected_collectives_per_iteration,
            "missing_collective_count": expected_collectives_per_iteration - observed_collectives,
        }
    return payload


def verify_seal(path: Path, model_version: str) -> dict[str, Any]:
    seal = json.loads(checked(path).read_text(encoding="utf-8"))
    expected = f"SEALED_BEFORE_{model_version.upper()}_EVALUATOR_ACCESS"
    if seal.get("status") != expected:
        raise ValueError(f"{model_version} prediction seal is invalid: {seal.get('status')} != {expected}")
    for artifact in seal["artifacts"]:
        source = checked(Path(artifact["path"]))
        if sha256(source) != artifact["sha256"] or source.stat().st_size != int(artifact["size_bytes"]):
            raise ValueError(f"sealed artifact changed: {source}")
    return seal


def build_payload(
    run: Path,
    model_version: str,
    manifest_path: Path,
    trace_root: Path,
    collective_events_path: Path,
    trace_metadata_path: Path,
    representative_lane_override: int | None = None,
) -> tuple[dict[str, Any], list[Path]]:
    seal_path = run / "predictions/prediction_seal.json"
    verify_seal(seal_path, model_version)
    nodes_path = checked(run / f"predictions/dag_{model_version}_nodes.csv.gz")
    edges_path = checked(run / f"predictions/dag_{model_version}_edges.csv.gz")
    contract_path = checked(run / "prediction_contract.json")
    truth_path = checked(run / "evaluator_only/iteration_ground_truth.csv")
    evaluation_path = checked(run / "evaluator_only/iteration_evaluation.csv")
    nodes = pd.read_csv(nodes_path, low_memory=False)
    critical_lanes = nodes.loc[
        nodes["on_critical_path"].astype(bool) & nodes["pp_lane"].ge(0), "pp_lane"
    ].astype(int)
    if critical_lanes.empty:
        raise ValueError(f"{model_version} critical path has no PP lane")
    representative_lane = (
        int(representative_lane_override)
        if representative_lane_override is not None
        else int(critical_lanes.mode().iloc[0])
    )
    if representative_lane not in range(16):
        raise ValueError(f"representative PP lane is outside 0--15: {representative_lane}")
    phase_nodes = nodes[
        nodes["rank"].ge(0)
        & nodes["phase"].isin(["FWD", "BWD"])
    ].copy()
    predicted_rank_phases = phase_nodes.groupby(
        ["rank", "pp_stage", "pp_lane", "phase", "microbatch"], as_index=False
    ).agg(
        start_ns=("predicted_start_ns", "min"), end_ns=("predicted_end_ns", "max"),
        critical=("on_critical_path", "any"), node_count=("node_id", "count"),
        compute_exposed_ns=("compute_exposed_ns_model", "sum"),
        compute_overlap_ns=("compute_overlap_ns_model", "sum"),
        network_ns=("network_service_ns_model", "sum"),
        software_sync_ns=("software_sync_ns_model", "sum"),
        framework_residual_ns=("framework_residual_ns_model", "sum"),
    )
    predicted = aggregate_phase_envelopes(
        predicted_rank_phases,
        start_col="start_ns",
        end_col="end_ns",
        stage_count=14,
        microbatch_count=3,
        include_iteration=False,
    ).rename(columns={"first_start_ns": "start_ns"})
    predicted_details = predicted_rank_phases.groupby(
        ["pp_stage", "phase", "microbatch"], as_index=False
    ).agg(
        critical=("critical", "any"), node_count=("node_count", "sum"),
        compute_exposed_ns=("compute_exposed_ns", "sum"),
        compute_overlap_ns=("compute_overlap_ns", "sum"),
        network_ns=("network_ns", "sum"),
        software_sync_ns=("software_sync_ns", "sum"),
        framework_residual_ns=("framework_residual_ns", "sum"),
    )
    predicted = predicted.merge(
        predicted_details,
        on=["pp_stage", "phase", "microbatch"],
        validate="one_to_one",
    )
    if len(predicted) != 14 * 6:
        raise ValueError(f"predicted PP14x3 phase grid incomplete: {len(predicted)}")
    predicted_origin = int(predicted["start_ns"].min())
    predicted["start_ms"] = (predicted["start_ns"] - predicted_origin) / 1e6
    predicted["end_ms"] = (predicted["end_ns"] - predicted_origin) / 1e6
    predicted["duration_ms"] = (predicted["end_ns"] - predicted["start_ns"]) / 1e6
    envelope_path: Path | None = None
    if model_version == "v67":
        envelope_path = checked(run / "predictions/phase_runtime_envelope.csv")
        envelope = pd.read_csv(envelope_path)
        envelope = envelope.groupby(
            ["pp_stage", "phase", "microbatch"], as_index=False
        ).agg(
            runtime_p10_duration_ns=("runtime_p10_duration_ns", "min"),
            runtime_p90_duration_ns=("runtime_p90_duration_ns", "max"),
        )
        predicted = predicted.merge(
            envelope,
            on=["pp_stage", "phase", "microbatch"],
            validate="one_to_one",
        )
    predicted_collectives = predicted_collective_bars(nodes, representative_lane, predicted_origin)
    representative_predicted_nodes = phase_nodes[phase_nodes["pp_lane"].eq(representative_lane)]
    boundaries = representative_predicted_nodes[
        representative_predicted_nodes["op_name"].isin(["fwd_start", "fwd_end", "bwd_start", "bwd_end"])
    ].copy()
    boundary_lookup = {
        (
            int(row.pp_stage), str(row.phase), int(row.microbatch),
            "start" if str(row.op_name).endswith("start") else "end",
        ): str(row.node_id)
        for row in boundaries.itertuples(index=False)
    }
    graph_edges = pd.read_csv(edges_path, low_memory=False)
    direct_edges = {
        (str(row.src), str(row.dst), str(row.edge_type))
        for row in graph_edges.itertuples(index=False)
    }
    phase_handoff_start = {
        (str(row.src), str(row.dst))
        for row in graph_edges[graph_edges["edge_type"].eq("phase_handoff_start")].itertuples(index=False)
    }
    phase_handoff_complete = {
        (str(row.src), str(row.dst))
        for row in graph_edges[graph_edges["edge_type"].eq("phase_handoff_complete")].itertuples(index=False)
    }
    release_start = {
        (str(row.src), str(row.dst))
        for row in graph_edges[graph_edges["edge_type"].eq("microbatch_release_start")].itertuples(index=False)
    }
    release_complete = {
        (str(row.src), str(row.dst))
        for row in graph_edges[graph_edges["edge_type"].eq("microbatch_release_complete")].itertuples(index=False)
    }
    dependencies = semantic_dependencies()
    for dependency in dependencies:
        source_match = re.fullmatch(r"s([0-9]+):([FB])([0-9]+)", dependency["source"])
        target_match = re.fullmatch(r"s([0-9]+):([FB])([0-9]+)", dependency["target"])
        if not source_match or not target_match:
            raise ValueError(f"bad dependency id: {dependency}")
        source_stage, source_symbol, source_mb = source_match.groups()
        target_stage, target_symbol, target_mb = target_match.groups()
        source_phase = "FWD" if source_symbol == "F" else "BWD"
        target_phase = "FWD" if target_symbol == "F" else "BWD"
        source_end = boundary_lookup[(int(source_stage), source_phase, int(source_mb), "end")]
        target_start = boundary_lookup[(int(target_stage), target_phase, int(target_mb), "start")]
        kind = dependency["kind"]
        if kind == "rank_program_order":
            verified = (source_end, target_start, "rank_program_order") in direct_edges
            if not verified and model_version in {"v64a", "v65", "v66", "v67"}:
                handoff_id = f"handoff:{source_end}__to__{target_start}"
                verified = (
                    (source_end, handoff_id) in phase_handoff_start
                    and (handoff_id, target_start) in phase_handoff_complete
                )
            if not verified and model_version == "v67":
                releases = [target for source, target in release_start if source == source_end]
                verified = any((release, target_start) in release_complete for release in releases)
        elif kind == "autograd_activation":
            verified = (source_end, target_start, "autograd_activation_dependency") in direct_edges
        elif kind == "pp_activation":
            pp_node = (
                f"pp:lane{representative_lane}:fwd{source_mb}:"
                f"s{source_stage}_to_s{target_stage}"
            )
            verified = (
                (source_end, pp_node, "pp_activation_send") in direct_edges
                and (pp_node, target_start, "pp_activation_recv") in direct_edges
            )
        elif kind == "pp_gradient":
            pp_node = (
                f"pp:lane{representative_lane}:bwd{source_mb}:"
                f"s{source_stage}_to_s{target_stage}"
            )
            verified = (
                (source_end, pp_node, "pp_gradient_send") in direct_edges
                and (pp_node, target_start, "pp_gradient_recv") in direct_edges
            )
        else:  # pragma: no cover
            raise KeyError(kind)
        dependency["verified_in_model_edges"] = bool(verified)
    if not all(bool(item["verified_in_model_edges"]) for item in dependencies):
        missing = [item for item in dependencies if not item["verified_in_model_edges"]]
        raise ValueError(f"{model_version} dependency audit failed: {missing[:3]}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    predicted_decomposition = phase_decomposition(
        float(contract["prediction"]["profiler_step_ms"]), predicted["start_ms"], predicted["end_ms"]
    )

    evaluator = run / "evaluator_only"
    manifest = pd.read_csv(checked(manifest_path))
    trace_events, target_phase_input_manifest, target_phase_caches = load_target_phase_envelopes(
        manifest, trace_root, evaluator
    )
    collective_cache = run / f"evaluator_only/timeline_trace_lane{representative_lane}_dp_edp_events_60_100.csv"
    trace_collectives = load_trace_collectives(
        checked(collective_events_path), checked(trace_metadata_path),
        collective_cache, representative_lane
    )
    source_256_trace = build_source_256_trace_payload(
        SOURCE_256_PHASE_EVENTS,
        SOURCE_256_COLLECTIVE_EVENTS,
        representative_lane,
    )
    truth = pd.read_csv(truth_path)
    truth = truth[truth["iteration"].isin(ITERATIONS)].copy()
    if len(truth) != len(ITERATIONS):
        raise ValueError("trace timeline truth grid incomplete")

    trace_payload: dict[str, Any] = {}
    stage_ratios: dict[str, Any] = {}
    predicted_stage_sum = predicted.groupby("pp_stage")["duration_ms"].sum()
    for iteration, frame in trace_events.groupby("iteration"):
        frame = frame.copy().sort_values(["first_start_ns", "pp_stage"])
        origin = int(frame["first_start_ns"].min())
        frame["start_ms"] = (frame["first_start_ns"] - origin) / 1e6
        frame["last_start_ms"] = (frame["last_start_ns"] - origin) / 1e6
        frame["end_ms"] = (frame["end_ns"] - origin) / 1e6
        frame["duration_ms"] = frame["duration_ns"] / 1e6
        profiler_ms = float(truth.loc[truth["iteration"].eq(iteration), "actual_profiler_step_ms"].iloc[0])
        decomposition = phase_decomposition(profiler_ms, frame["start_ms"], frame["end_ms"])
        collective_frame = trace_collectives[trace_collectives["iteration"].eq(iteration)].copy()
        collective_frame["start_ms"] = (collective_frame["trace_start_ns"] - origin) / 1e6
        collective_frame["end_ms"] = (collective_frame["trace_end_ns"] - origin) / 1e6
        collective_bars = []
        for row in collective_frame.itertuples(index=False):
            bar = {
                "id": str(row.event_id), "stage": int(row.pp_stage),
                "rank": int(row.rank), "group_type": str(row.group_type),
                "collective": str(row.collective),
                "start_ms": round(float(row.start_ms), 6),
                "end_ms": round(float(row.end_ms), 6),
                "duration_ms": round(float(row.duration_ns) / 1e6, 6),
                "payload_mib": round(float(row.logical_input_bytes) / 1048576, 3),
            }
            if str(row.group_type) == "dp" and str(row.collective) == "rs":
                first_release_ms = (
                    float(row.dp_group_first_release_trace_ns) - origin
                ) / 1e6
                last_release_ms = (
                    float(row.dp_group_last_release_trace_ns) - origin
                ) / 1e6
                group_end_ms = (
                    float(row.dp_group_end_trace_ns) - origin
                ) / 1e6
                bar.update({
                    "rank_start_ms": bar["start_ms"],
                    "rank_end_ms": bar["end_ms"],
                    "rank_duration_ms": bar["duration_ms"],
                    "group_key": str(row.dp_group_key),
                    "group_observed_rank_count": int(row.dp_group_observed_rank_count),
                    "group_expected_rank_count": int(row.dp_group_expected_rank_count),
                    "start_ms": round(first_release_ms, 6),
                    "last_release_ms": round(last_release_ms, 6),
                    "end_ms": round(group_end_ms, 6),
                    "duration_ms": round(group_end_ms - first_release_ms, 6),
                    "early_residence_ms": round(
                        max(0.0, min(float(row.end_ms), last_release_ms) - float(row.start_ms)),
                        6,
                    ),
                    "rank_post_last_release_ms": round(
                        max(0.0, float(row.end_ms) - max(float(row.start_ms), last_release_ms)),
                        6,
                    ),
                    "group_arrival_span_ms": round(
                        float(row.dp_group_arrival_span_ns) / 1e6, 6
                    ),
                    "group_tail_after_last_release_ms": round(
                        float(row.dp_group_tail_after_last_release_ns) / 1e6, 6
                    ),
                })
            collective_bars.append(bar)
        collective_span = {
            "first_start_ms": round(float(collective_frame["start_ms"].min()), 6),
            "last_end_ms": round(float(collective_frame["end_ms"].max()), 6),
            "span_ms": round(
                float(collective_frame["end_ms"].max() - collective_frame["start_ms"].min()), 6
            ),
        }
        trace_payload[str(int(iteration))] = {
            "bars": [
                {
                    "id": f"s{int(row.pp_stage)}:{'F' if str(row.phase) == 'forward' else 'B'}{int(row.microbatch)}",
                    "stage": int(row.pp_stage), "phase": str(row.phase), "microbatch": int(row.microbatch),
                    "start_ms": round(float(row.start_ms), 6), "end_ms": round(float(row.end_ms), 6),
                    "duration_ms": round(float(row.duration_ms), 6),
                    "last_start_ms": round(float(row.last_start_ms), 6),
                    "active_union_ms": round(float(row.active_union_ns) / 1e6, 6),
                    "rank_count": int(row.rank_count),
                }
                for row in frame.itertuples(index=False)
            ],
            "collectives": collective_bars,
            "collective_span": collective_span,
            "decomposition": {key: round(value, 6) for key, value in decomposition.items()},
        }
        trace_stage_sum = frame.groupby("pp_stage")["duration_ms"].sum()
        stage_ratios[str(int(iteration))] = [
            {
                "stage": int(stage),
                "predicted_operation_sum_ms": round(float(predicted_stage_sum.loc[stage]), 6),
                "trace_operation_sum_ms": round(float(trace_stage_sum.loc[stage]), 6),
                "trace_over_predicted": round(float(trace_stage_sum.loc[stage] / predicted_stage_sum.loc[stage]), 6),
            }
            for stage in range(14)
        ]
    evaluation = pd.read_csv(evaluation_path)
    evaluation = evaluation[evaluation["split"].eq("validation")].sort_values("iteration")
    model_access_path = checked(run / "input_access_audit.json")
    evaluation_metrics_path = checked(run / "evaluator_only/metrics.json")
    model_access = json.loads(model_access_path.read_text(encoding="utf-8"))
    evaluation_metrics = json.loads(evaluation_metrics_path.read_text(encoding="utf-8"))
    direct_target_timing_files = int(model_access.get("target_timing_files_read", -1))
    if direct_target_timing_files != 0:
        raise ValueError(f"{model_version} model process read target timing files")
    evaluation_scope = str(evaluation_metrics.get("scope", ""))
    if "retrospective" not in evaluation_scope or "not untouched validation" not in evaluation_scope:
        raise ValueError("224-GPU evaluation scope no longer records retrospective development use")
    predicted_bars = [
        {
            "id": f"s{int(row.pp_stage)}:{'F' if str(row.phase) == 'FWD' else 'B'}{int(row.microbatch)}",
            "stage": int(row.pp_stage),
            "phase": "forward" if str(row.phase) == "FWD" else "backward",
            "microbatch": int(row.microbatch), "start_ms": round(float(row.start_ms), 6),
            "end_ms": round(float(row.end_ms), 6), "duration_ms": round(float(row.duration_ms), 6),
            "last_start_ms": round((float(row.last_start_ns) - predicted_origin) / 1e6, 6),
            "active_union_ms": round(float(row.active_union_ns) / 1e6, 6),
            "rank_count": int(row.rank_count),
            "critical": bool(row.critical), "node_count": int(row.node_count),
            "compute_exposed_ms": round(float(row.compute_exposed_ns) / 1e6, 6),
            "compute_overlap_ms": round(float(row.compute_overlap_ns) / 1e6, 6),
            "network_ms": round(float(row.network_ns) / 1e6, 6),
            "software_sync_ms": round(float(row.software_sync_ns) / 1e6, 6),
            "framework_residual_ms": round(float(row.framework_residual_ns) / 1e6, 6),
        }
        for row in predicted.itertuples(index=False)
    ]
    if model_version == "v67":
        envelope_lookup = {
            (int(row.pp_stage), str(row.phase), int(row.microbatch)): row
            for row in predicted.itertuples(index=False)
        }
        for item in predicted_bars:
            phase = "FWD" if item["phase"] == "forward" else "BWD"
            row = envelope_lookup[(item["stage"], phase, item["microbatch"])]
            item["runtime_p10_duration_ms"] = round(float(row.runtime_p10_duration_ns) / 1e6, 6)
            item["runtime_p90_duration_ms"] = round(float(row.runtime_p90_duration_ns) / 1e6, 6)
    predicted_gap_rows = adjacent_phase_gaps(pd.DataFrame(predicted_bars))
    trace_gap_rows = {
        iteration: adjacent_phase_gaps(pd.DataFrame(value["bars"]))
        for iteration, value in trace_payload.items()
    }
    tail_collectives = [
        item for item in predicted_collectives if item["group_type"] != "handoff"
    ]
    payload = {
        "schema": f"dag-{display_version(model_version)}-predicted-vs-trace-global-timeline-v3",
        "model_version": model_version,
        "model_label": f"DAG {display_version(model_version)}",
        "representative_lane": representative_lane,
        "iterations": list(ITERATIONS), "default_iteration": 100,
        "predicted": {
            "bars": predicted_bars,
            "collectives": predicted_collectives,
            "collective_span": {
                "first_start_ms": round(min(item["start_ms"] for item in tail_collectives), 6),
                "last_end_ms": round(max(item["end_ms"] for item in tail_collectives), 6),
                "span_ms": round(
                    max(item["end_ms"] for item in tail_collectives)
                    - min(item["start_ms"] for item in tail_collectives), 6
                ),
            },
            "dependencies": dependencies,
            "decomposition": {key: round(value, 6) for key, value in predicted_decomposition.items()},
            "training_ms": round(float(contract["prediction"]["training_step_ms"]), 6),
            "raw_graph_ms": round(float(contract["prediction"]["raw_graph_ms"]), 6),
        },
        "trace": trace_payload,
        "source_256_trace": source_256_trace,
        "stage_ratios": stage_ratios,
        "adjacent_phase_gaps": {
            "predicted": predicted_gap_rows,
            "trace": trace_gap_rows,
            "semantics": "next 16-rank phase-envelope start minus previous 16-rank phase-envelope end",
        },
        "accuracy": {
            "iteration": evaluation["iteration"].astype(int).tolist(),
            "actual_profiler_ms": evaluation["actual_profiler_step_ms"].astype(float).round(6).tolist(),
            "predicted_profiler_ms": evaluation["predicted_profiler_step_ms"].astype(float).round(6).tolist(),
        },
        "visual_contract": {
            "phase_colors": PHASE_COLORS,
            "phase_tracks": PHASE_TRACKS,
            "shared_by": ["predicted", "target_224_trace", "source_256_trace"],
            "critical_path_style": "top border only; phase fill and track are unchanged",
        },
        "html_evolution": [
            {"version": version, "change": change} for version, change in HTML_EVOLUTION
        ],
        "data_scope_audit": {
            "status": "RETROSPECTIVE_DEVELOPMENT_SET_NOT_BLIND_VALIDATION",
            "model_process_target_timing_files_read": direct_target_timing_files,
            "model_process_target_fields_read": model_access.get("target_fields_read", []),
            "model_source_trace_path": str(SOURCE_256_PHASE_EVENTS.resolve()),
            "model_source_iterations": model_access.get("source_iterations", []),
            "target_trace_role": "evaluator_only_visualization_and_retrospective_diagnostics",
            "target_trace_files_read_by_evaluator": len(ITERATIONS) * 224,
            "method_development_target_diagnostics_seen": True,
            "formal_224_accuracy_claim_allowed": False,
            "evaluation_scope": evaluation_scope,
            "phase_bar_granularity": "one PP stage x one phase x one microbatch x 16 ranks envelope",
        },
        "semantics": {
            "alignment": "All three timelines set the earliest start among all 16-rank FWD phase envelopes to t=0 and share one displayed scale.",
            "phase_bars": "A/B/C each render one bar per PP stage, phase and microbatch: earliest of 16 rank starts to latest of 16 rank completions; active union and last-rank start remain in the tooltip.",
            "trace_bars": "Kineto user_annotation forward_step/backward_step grouped across all 16 ranks of each PP stage from 224-GPU target traces.",
            "trace_collectives": "Selected-rank Kineto GPU collective kernels for DP/Expert-DP RS, AG0 and AG1; evaluator-only target observations.",
            "source_256_trace": "Source-calibration PP16xM4 F/B annotations grouped across all 16 ranks, plus selected-rank DP/Expert-DP RS, AG0 and AG1 events from iterations 60-100.",
            "predicted_collectives": f"{model_version} group collective_service nodes; duration separates network_service and post-arrival software_sync in the tooltip.",
            "phase_handoff": "Source-trace lower-envelope F-to-B and B-to-F scheduler handoff nodes remain in the model; v6.7-ui3 hides all handoff bars in the main prediction timeline so A/B/C use the same visible event classes.",
            "trace_dp_rs_split": "For B/C DP RS, the visible bar uses the same group key and full Trace group envelope as dp_stage_mfu_model.html: first rank start to last rank start is arrival, then last rank start to group end is service. The representative-rank kernel remains tooltip-only.",
            "microbatch_release": "v6.7: on the same rank, the source-trace median from the previous F/B end to the next F/B start is an adjacent-F/B start interval in max-plus scheduling; PP/autograd dependencies can impose a later start.",
            "predicted_dp_rs_arrival": "A renders the existing per-rank collective release-node envelope. It is already part of DAG scheduling and is not added again after the last rank arrives.",
            "runtime_band": "v6.7-ui1: source256 phase-local p10-p90 duration data is retained in the payload but is not rendered on the main timeline.",
            "outside_phase_envelope": "Profiler Step minus first-FWD-to-last-BWD phase envelope; displayed after the envelope only as an accounting ribbon, not an observed physical placement.",
            "critical": f"Yellow outline marks {model_version} nodes on the predicted max-plus critical path.",
            "dependencies": f"Dependency arrows are verified against {model_version} rank_program_order, autograd and PP send/recv edges; trace arrows reuse the same code semantics and are not separately observed events.",
        },
    }
    inputs = [
        nodes_path, edges_path, contract_path, truth_path, evaluation_path,
        checked(manifest_path), checked(collective_events_path),
        checked(trace_metadata_path), checked(SOURCE_256_PHASE_EVENTS),
        checked(SOURCE_256_COLLECTIVE_EVENTS), seal_path, model_access_path,
        evaluation_metrics_path, target_phase_input_manifest, *target_phase_caches,
    ]
    if envelope_path is not None:
        inputs.append(envelope_path)
    verify_seal(seal_path, model_version)
    return payload, inputs


def render_html(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    label = payload["model_label"]
    version = payload["model_version"]
    evolution_rows = "".join(
        f"<tr><td><code>{item['version']}</code></td><td>{item['change']}</td></tr>"
        for item in payload["html_evolution"]
    )
    if version in {"v66", "v67"}:
        tail_badge = f"{display_version(version)} optimizer 分段"
        tail_order = "DP RS → EDP RS → world AR0 → optimizer小段 → world AR1 → optimizer小段 → DP AG0 → EDP AG0 → optimizer主体/组间重叠 → stage join → DP AG1 → EDP AG1"
        tail_note = "AG0→AG1 不再是一整段固定等待；optimizer 主体由各 Expert-DP 组完成后分别释放，最后只做零时长依赖汇合。"
    else:
        tail_badge = "v6.3.1+ 显式barrier" if version in {"v631", "v64a", "v65"} else ("v6.3 已修正" if version == "v63" else "v6.2 旧语义")
        tail_order = "DP RS → EDP RS → 全局 RS 完成 → optimizer → DP AG0 → EDP AG0 → stage AG barrier → DP AG1 → EDP AG1"
        tail_note = "RS 与 AG 都不是 DP、EDP 同时启动；AG 第一轮与第二轮之间还有同 PP stage 的显式同步 barrier。"
    predicted_bars = {item["id"]: item for item in payload["predicted"]["bars"]}
    pp0_f2b_gap_s = (
        predicted_bars["s0:B0"]["start_ms"] - predicted_bars["s0:F2"]["end_ms"]
    ) / 1000.0
    if version == "v67":
        critical_handoffs = [
            item for item in payload["predicted"]["collectives"]
            if item["group_type"] == "handoff" and item["critical"]
        ]
        if not critical_handoffs:
            raise ValueError("v6.7 critical path has no phase handoff")
        dominant_handoff = max(critical_handoffs, key=lambda item: item["duration_ms"])
        runtime_note = (
        "<div class=\"notice\"><b>相邻 F/B 启动间隔：</b>这里的对象很具体——同一个rank（一个GPU训练进程）中，上一段F/B annotation结束，"
        "到下一段F/B annotation开始之间的时间差；它不表示释放通信、显存或其他资源。对每个rank、映射后的PP位置和F→F/B→B/F→B/B→F转移，"
        "从256卡iter 60–100取非负中位数。DAG下一段开始时间为"
        "<code>max(上一段结束+该间隔, PP输入到达, autograd依赖完成, 其他前驱完成)</code>；整个预测过程不读取224卡时间。"
        f"当前 PP0 的F2→B0可见空档约 <b>{pp0_f2b_gap_s:.3f}s</b>，它是等待的近似表示，不是一个长计算算子。</div>"
        "<div class=\"notice\"><b>为什么黄色关键路径只经过PP0–PP3：</b>黄色上沿表示lane"
        f"{payload['representative_lane']}的单条max-plus最长前驱链。它在PP{dominant_handoff['stage']}的F2后走入约"
        f" <b>{dominant_handoff['duration_ms']/1000.0:.3f}s</b> 的Trace派生F→B启动间隔节点，再沿梯度链返回PP0；"
        "PP4–PP13分支在该约束满足前已经完成，存在slack，因而没有黄色上沿。这是当前粗粒度就绪节点掩盖下游PP依赖的模型诊断，"
        "并不表示PP4–PP13没有工作或不重要。</div>"
        )
    else:
        runtime_note = ""
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{label} · 224卡预测/Trace与256卡源Trace</title>
<style>
:root{{--bg:#07111d;--panel:#0e1b2c;--panel2:#101f33;--line:#2b4059;--text:#e9f2fb;--muted:#93a8bd;--fwd:{PHASE_COLORS['forward']};--bwd:{PHASE_COLORS['backward']};--dp-rs:#27d17f;--edp-rs:#00b8a9;--dp-ag:#ffb54a;--edp-ag:#b78cff;--critical:#ffe45e;--trace:#55d6be;--residual:#f59e0b;--danger:#ff6b6b}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 Inter,system-ui,sans-serif}}main{{max-width:1580px;margin:auto;padding:28px}}h1{{font-size:34px;margin:4px 0 8px}}h2{{font-size:20px;margin:0 0 5px}}p{{margin:5px 0}}.muted{{color:var(--muted)}}.eyebrow{{color:var(--trace);font-weight:800;letter-spacing:.12em;font-size:12px}}.controls{{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0}}button{{border:1px solid var(--line);background:var(--panel);color:var(--muted);padding:7px 11px;border-radius:6px;cursor:pointer}}button.active{{color:#fff;border-color:var(--trace);background:#163b43}}.cards{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;margin:14px 0}}.card,.panel{{background:var(--panel);border:1px solid var(--line)}}.card{{padding:14px}}.card .v{{font-size:23px;font-weight:800;margin-top:5px;font-variant-numeric:tabular-nums}}.card .k{{color:var(--muted);font-size:12px}}.panel{{padding:17px;margin:12px 0}}.head{{display:flex;justify-content:space-between;gap:18px;align-items:flex-start}}.badge{{border:1px solid var(--line);padding:4px 8px;color:var(--muted);font-size:12px}}.legend{{display:flex;gap:16px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin:10px 0}}.sw{{display:inline-block;width:17px;height:9px;margin-right:5px;vertical-align:middle}}.timeline{{overflow-x:auto;background:#081523;border:1px solid #22384f;padding:5px}}.timeline svg{{display:block;min-width:1350px;width:100%;height:auto}}.two{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}.ribbon{{height:34px;display:flex;border:1px solid var(--line);margin-top:10px;overflow:hidden}}.ribbon span{{display:grid;place-items:center;min-width:1px;white-space:nowrap;overflow:hidden;font-size:11px}}.phase{{background:#245a85}}.outside{{background:#9a6211}}.ratio-chart{{display:grid;grid-template-columns:repeat(14,minmax(38px,1fr));gap:7px;height:280px;align-items:end;padding-top:28px;border-bottom:1px solid var(--line)}}.ratio-col{{height:100%;display:flex;flex-direction:column;justify-content:flex-end;align-items:center}}.ratio-v{{font-size:11px;color:var(--muted)}}.ratio-bar-wrap{{height:190px;width:70%;display:flex;align-items:flex-end;position:relative;border-bottom:1px solid #597087}}.ratio-one{{position:absolute;left:-15%;right:-15%;bottom:50%;border-top:1px dashed #e8edf5;opacity:.6}}.ratio-bar{{width:100%;background:var(--trace);border-top:2px solid #a4ffea}}.ratio-bar.hot{{background:var(--danger);border-color:#ffc0c0}}.ratio-stage{{font-size:11px;color:var(--muted);margin-top:5px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:8px;border-bottom:1px solid var(--line);text-align:right;font-variant-numeric:tabular-nums}}th:first-child,td:first-child{{text-align:left}}th{{font-size:11px;color:var(--muted)}}.notice{{border-left:4px solid var(--residual);padding:12px 14px;background:#191d28;margin:12px 0}}code{{color:#9cecff}}.dep-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:10px}}.dep-chain{{border:1px solid var(--line);background:#091727;padding:12px}}.dep-chain strong{{display:block;margin-bottom:6px}}.dep-chain code{{word-spacing:3px}}@media(max-width:900px){{.cards{{grid-template-columns:1fr 1fr}}.two,.dep-grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<div class="eyebrow">{label} · source calibration + evaluator-only visualization</div><h1>224卡预测/Trace与256卡源Trace：全局时序对照</h1>
<p class="muted">三幅图的F/B均按“一个PP stage × 一个microbatch × 16个rank”的包络绘制，并共用横轴尺度；切换iteration可对照256卡来源和224卡回顾性误差。</p>
<div class="controls" id="iterationControls"></div>
<div class="controls" id="dependencyControls"><span class="muted" style="padding:7px 0">依赖箭头：</span><button data-dep="none">隐藏</button><button data-dep="late">PP11–PP13</button><button data-dep="all">全部stage</button></div>
<div class="cards"><div class="card"><div class="k">DAG Profiler</div><div class="v" id="predProfiler"></div></div><div class="card"><div class="k">Trace Profiler</div><div class="v" id="traceProfiler"></div></div><div class="card"><div class="k">Profiler 总缺口</div><div class="v" id="profGap"></div></div><div class="card"><div class="k">F/B 包络缺口</div><div class="v" id="phaseGap"></div></div><div class="card"><div class="k">包络外时间缺口</div><div class="v" id="outsideGap"></div></div></div>
<div class="legend"><span><i class="sw" style="background:var(--fwd)"></i>FWD（预测/Trace同色）</span><span><i class="sw" style="background:var(--bwd)"></i>BWD（预测/Trace同色）</span><span><i class="sw" style="background:var(--dp-rs)"></i>DP RS service（ready→group end）</span><span><i class="sw" style="background:#64748b"></i>DP RS arrival（first→last rank）</span><span><i class="sw" style="background:var(--edp-rs)"></i>EDP RS</span><span><i class="sw" style="background:var(--dp-ag)"></i>DP AG</span><span><i class="sw" style="background:var(--edp-ag)"></i>EDP AG</span><span><i class="sw" style="background:#ff66c4"></i>optimizer分段</span><span><i class="sw" style="background:#f4d35e"></i>AG轮次barrier</span><span><i class="sw" style="border-top:2px solid var(--critical)"></i>DAG关键路径上沿</span><span style="color:#ffb454">橙线：rank内1F1B顺序</span><span style="color:#55d6be">青线：PP activation</span><span style="color:#ef79ff">紫线：PP gradient</span><span style="color:#ffe478">黄虚线：F→B autograd</span></div>
<section class="panel"><div class="head"><div><h2>最后三个stage的依赖顺序</h2><p class="muted">不是F0/F1/F2彼此独立；stage越靠后，越早在forward之间插入对应backward。</p></div><span class="badge">已从{version} edges逐边验证</span></div><div class="dep-grid"><div class="dep-chain"><strong>PP11 · 首个B之前有3个F</strong><code>F0 → F1 → F2 → B0 → B1 → B2</code></div><div class="dep-chain"><strong>PP12 · 首个B之前有2个F</strong><code>F0 → F1 → B0 → F2 → B1 → B2</code></div><div class="dep-chain"><strong>PP13 · 首个B之前有1个F</strong><code>F0 → B0 → F1 → B1 → F2 → B2</code></div></div><div class="notice">每个Fm还要等待上一个stage发来的同microbatch activation；每个Bm还要等待下一个stage返回的同microbatch gradient。因此节点开始时间取多个前驱完成时间的最大值。</div></section>
<section class="panel"><div class="head"><div><h2>Optimizer tail 的 DP / EDP 顺序</h2><p class="muted">{tail_note}</p></div><span class="badge">{tail_badge}</span></div><div class="notice"><code>{tail_order}</code></div></section>
<section class="panel"><div class="head"><div><h2>A · {version} 预测全局时序</h2><p class="muted">PP14 × 3；每个F/B粗条覆盖该stage全部16个预测rank，从最早开始到最晚完成。依赖箭头和黄色关键路径用lane{payload['representative_lane']}逐边核验。DP RS灰段为16个rank的预测到达跨度，绿段为最后rank到达后的group service。</p></div><span class="badge">冻结预测</span></div>{runtime_note}<div class="timeline" id="predictedTimeline"></div><div id="predictedRibbon"></div></section>
<section class="panel"><div class="head"><div><h2>B · 224 卡 Trace 全局时序</h2><p class="muted">每个F/B粗条由该PP stage全部16个rank的user annotation组成：最早rank开始到最晚rank完成。DP RS也使用完整16-rank group envelope；其余DP/EDP细条仍只作代表rank诊断。</p></div><span class="badge" id="traceBadge"></span></div><div class="timeline" id="traceTimeline"></div><div id="traceRibbon"></div></section>
<section class="panel"><div class="head"><div><h2>C · 256 卡源 Trace 全局时序</h2><p class="muted">PP16 × 4 microbatch；F/B与B图完全同口径，均按每stage全部16个rank聚合。DP RS同样使用完整16-rank group envelope。数据来自v6.7源校准Trace，不补造缺失事件。</p></div><span class="badge" id="sourceBadge"></span></div><div class="timeline" id="sourceTimeline"></div></section>
<section class="panel"><div class="head"><div><h2>DP RS group envelope分解</h2><p class="muted">对全部PP stage取中位数。A/B/C统一显示：灰段为组内最早到最晚Rank就绪，绿段为最后Rank就绪到全组完成；A是预测，B/C是实测。</p></div><span class="badge">与DP阶段页面同口径</span></div><table id="dpRsTable"></table><div class="notice">arrival段是Rank就绪差异在DAG中的自然结果，不是在最后Rank到达后再额外串行增加的一段时间。A图现在把已有的到达跨度画出来，但没有修改模型总时长；绿色service才对应集合通信完成阶段。</div><p class="muted">统一定义：<code>arrival = ready - start</code>，<code>service = end - ready</code>，<code>group FCT = end - start</code>。跨16/224/256卡case的P10/P50/P90概览见 <a href="/case_256gpu_pp16_cp2_a2a/figures/dp_stage_mfu_model.html" target="_blank">DP阶段MFU模型页面</a>；实测表显示当前选中iteration，A为冻结预测。</p></section>
<section class="panel"><div class="head"><div><h2>按 PP stage 聚合的 F/B 包络倍率</h2><p class="muted">每个stage的6个16-rank F/B包络时长之和：Trace ÷ DAG。1.0表示预测一致，红色表示高于1.2。</p></div></div><div class="ratio-chart" id="ratioChart"></div></section>
<section class="panel"><div class="head"><div><h2>相邻 microbatch 的包络间隔</h2><p class="muted">后一个16-rank F/B包络的最早开始，减去前一个包络的最晚完成。负数表示两个microbatch在不同rank上存在交叠；它用于同口径观察，不等同于模型内部逐rank的“相邻F/B启动间隔”。</p></div><span class="badge">单位 ms</span></div><table id="gapTable"></table></section>
<section class="panel two"><div><h2>选中 iteration 的224卡差异分解</h2><table id="differenceTable"></table></div><div><h2>怎样阅读</h2><div class="notice">三幅图都把全部F/B包络中的最早FWD对齐到t=0，并共用横轴尺度。粗条不是某个代表GPU，而是16个rank的起止包络；tooltip同时给出最晚rank启动和区间并集。</div><p>A/B/C的DP RS也显示group envelope：A来自DAG的rank就绪节点，B/C来自Trace；灰色arrival与绿色service合起来才是group FCT。其他DP/EDP细条仍是代表rank kernel。</p><p class="muted">v6.7预测代码直接读取的时序数据只有256卡60–100；224卡时序只在预测封存后的evaluator中读取。但v6系列开发过程看过224卡诊断，所以这些指标只能作为回顾性开发集结果，不能当作从未看过224数据的盲测精度。</p></div></section>
<section class="panel"><div class="head"><div><h2>224 卡数据边界审计</h2><p class="muted">区分“预测进程是否直接读取目标时序”和“方法设计是否曾参考目标诊断”。</p></div><span class="badge">RETROSPECTIVE · 非盲测</span></div><div class="notice"><b>代码读取：</b>v6.7预测进程读取224卡时序文件数为 <code>{payload['data_scope_audit']['model_process_target_timing_files_read']}</code>；用于建模的F/B时序来自256卡iter 60–100。</div><div class="notice"><b>方法开发：</b>v6.5–v6.7设计已经参考过224卡诊断，因此当前224卡结果只能用于定位误差，不能用于正式外推胜负或宣称盲测精度。B图的224卡全部rank数据只进入evaluator-only页面。</div></section>
<section class="panel"><div class="head"><div><h2>HTML 演进日志</h2><p class="muted">记录可视化语义变化，不表示每一版都重新拟合模型。</p></div><span class="badge">更新至 {payload['html_evolution'][-1]['version']}</span></div><table><thead><tr><th>版本</th><th style="text-align:left">页面变化</th></tr></thead><tbody>{evolution_rows}</tbody></table><div class="notice">当前共享视觉合同：FWD <code>{PHASE_COLORS['forward']}</code> 位于上轨，BWD <code>{PHASE_COLORS['backward']}</code> 位于下轨，两轨相差 14 px；预测和 Trace 完全共用。关键路径只增加黄色上沿，不改变颜色或轨道。</div></section>
<script>const DATA={data};
const $=id=>document.getElementById(id),fmt=(v,d=3)=>Number(v).toFixed(d),ns='http://www.w3.org/2000/svg',PHASE_COLORS=DATA.visual_contract.phase_colors,PHASE_TRACKS=DATA.visual_contract.phase_tracks;let selected=String(DATA.default_iteration),depMode='late';
function add(svg,tag,attrs,text){{const e=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;svg.appendChild(e);return e}}
function selectedDependencies(){{if(depMode==='none')return[];if(depMode==='all')return DATA.predicted.dependencies;return DATA.predicted.dependencies.filter(d=>Math.max(Number(d.source.match(/^s(\\d+)/)[1]),Number(d.target.match(/^s(\\d+)/)[1]))>=11)}}
function timeline(target,bars,collectives,totalMs,critical,traceSemantic=false){{const W=1480,rowH=44,m={{l:58,r:22,t:28,b:38}},H=m.t+14*rowH+m.b,x=v=>m.l+(W-m.l-m.r)*v/totalMs;const svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox',`0 0 ${{W}} ${{H}}`);const defs=add(svg,'defs',{{}});for(const [kind,color] of Object.entries({{rank_program_order:'#ffb454',pp_activation:'#55d6be',pp_gradient:'#ef79ff',autograd_activation:'#ffe478'}})){{const marker=add(defs,'marker',{{id:`arrow-${{kind}}-${{traceSemantic?'trace':'dag'}}`,markerWidth:7,markerHeight:7,refX:6,refY:3,orient:'auto',markerUnits:'strokeWidth'}});add(marker,'path',{{d:'M0,0 L0,6 L6,3 z',fill:color}})}}for(let s=0;s<14;s++){{const y=m.t+s*rowH;add(svg,'rect',{{x:m.l,y,width:W-m.l-m.r,height:rowH,fill:s%2?'#0a1827':'#0c1b2c'}});add(svg,'text',{{x:m.l-8,y:y+25,'text-anchor':'end',fill:'#93a8bd','font-size':11}},`PP${{s}}`)}}for(let t=0;t<=totalMs;t+=5000){{const xx=x(t);add(svg,'line',{{x1:xx,y1:m.t-7,x2:xx,y2:H-m.b,stroke:'#294057','stroke-dasharray':'3 4'}});add(svg,'text',{{x:xx,y:H-13,'text-anchor':'middle',fill:'#93a8bd','font-size':10}},`${{fmt(t/1000,0)}}s`)}}const positions=new Map();bars.forEach(d=>{{const yy=m.t+d.stage*rowH+(d.phase==='forward'?4:16),h=10,w=Math.max(1.5,x(d.end_ms)-x(d.start_ms));positions.set(d.id,{{start:x(d.start_ms),end:x(d.end_ms),center:(x(d.start_ms)+x(d.end_ms))/2,y:yy+h/2,stage:d.stage}});const r=add(svg,'rect',{{x:x(d.start_ms),y:yy,width:w,height:h,rx:2,fill:d.phase==='forward'?'#24c8ff':'#ff725c',stroke:critical&&d.critical?'#ffe45e':'none','stroke-width':critical&&d.critical?2.2:0}});const title=document.createElementNS(ns,'title');title.textContent=`PP${{d.stage}} ${{d.phase==='forward'?'F':'B'}}${{d.microbatch}} · 16-rank envelope · ${{fmt(d.start_ms/1000,3)}}–${{fmt(d.end_ms/1000,3)}}s · wall ${{fmt(d.duration_ms,2)}}ms · last rank starts ${{fmt(d.last_start_ms/1000,3)}}s · active union ${{fmt(d.active_union_ms,2)}}ms`+(d.node_count?` · ${{d.node_count}} internal nodes`:``);r.appendChild(title);if(w>27)add(svg,'text',{{x:x(d.start_ms)+3,y:yy+8,fill:'#07111d','font-size':7,'font-weight':800}},`${{d.phase==='forward'?'F':'B'}}${{d.microbatch}}`)}});collectives.forEach(d=>{{const isDp=d.group_type==='dp',isOpt=d.group_type==='optimizer',isRs=d.collective==='rs',yy=m.t+d.stage*rowH+(isOpt?27:(isDp?31:37)),h=isOpt?3:5,w=Math.max(1.4,x(d.end_ms)-x(d.start_ms)),fill=isOpt?'#ff66c4':(isDp?(isRs?'#27d17f':'#ffb54a'):(isRs?'#00b8a9':'#b78cff'));const r=add(svg,'rect',{{x:x(d.start_ms),y:yy,width:w,height:h,rx:1,fill,stroke:critical&&d.critical?'#ffe45e':'none','stroke-width':critical&&d.critical?1.5:0}});const splitDpRs=traceSemantic&&isDp&&isRs&&Number.isFinite(d.last_release_ms);let earlyRect=null;if(splitDpRs){{const earlyEnd=Math.min(d.end_ms,Math.max(d.start_ms,d.last_release_ms));if(earlyEnd>d.start_ms)earlyRect=add(svg,'rect',{{x:x(d.start_ms),y:yy,width:Math.max(1.4,x(earlyEnd)-x(d.start_ms)),height:h,rx:1,fill:'#64748b'}})}}const title=document.createElementNS(ns,'title');const scope=splitDpRs?'Trace DP group envelope':(traceSemantic?`Trace rank${{d.rank}} GPU kernel`:(isOpt?'DAG rank optimizer segment':'DAG group service')),groupLabel=isOpt?'OPT':(d.group_type==='dp'?'DP':'EDP');title.textContent=`${{scope}} · PP${{d.stage}} ${{groupLabel}} ${{d.collective.toUpperCase()}} · ${{fmt(d.start_ms/1000,3)}}–${{fmt(d.end_ms/1000,3)}}s · ${{fmt(d.duration_ms,2)}}ms`+(splitDpRs?` · arrival ${{fmt(d.group_arrival_span_ms,2)}}ms · service ${{fmt(d.group_tail_after_last_release_ms,2)}}ms · rank${{d.rank}} kernel ${{fmt(d.rank_duration_ms,2)}}ms`:``)+(d.network_ms!==undefined?` · network ${{fmt(d.network_ms,2)}}ms · software ${{fmt(d.software_sync_ms,2)}}ms`:``)+(d.payload_mib!==undefined?` · ${{fmt(d.payload_mib,1)}} MiB`:``);r.appendChild(title);if(earlyRect){{const earlyTitle=document.createElementNS(ns,'title');earlyTitle.textContent=`DP RS group arrival ${{fmt(d.group_arrival_span_ms,2)}}ms：最早rank到最晚rank；期间可能已有部分通信`;earlyRect.appendChild(earlyTitle)}}}});const style={{rank_program_order:['#ffb454',''],pp_activation:['#55d6be',''],pp_gradient:['#ef79ff',''],autograd_activation:['#ffe478','5 3']}};selectedDependencies().forEach(d=>{{const a=positions.get(d.source),b=positions.get(d.target);if(!a||!b)return;const [color,dash]=style[d.kind],opacity=traceSemantic?.42:.72;let path;if(d.kind==='rank_program_order'){{const lift=a.y===b.y?a.y-10:(a.y+b.y)/2-7;path=`M ${{a.center}} ${{a.y}} Q ${{(a.center+b.center)/2}} ${{lift}} ${{b.center}} ${{b.y}}`}}else{{const mid=(a.end+b.start)/2;path=`M ${{a.end}} ${{a.y}} C ${{mid}} ${{a.y}}, ${{mid}} ${{b.y}}, ${{b.start}} ${{b.y}}`}}const e=add(svg,'path',{{d:path,fill:'none',stroke:color,'stroke-width':1.4,'stroke-opacity':opacity,'stroke-dasharray':dash,'marker-end':`url(#arrow-${{d.kind}}-${{traceSemantic?'trace':'dag'}})`,'pointer-events':'stroke'}});const title=document.createElementNS(ns,'title');title.textContent=(traceSemantic?'语义推断 · ':DATA.model_version+' edge已验证 · ')+d.label;e.appendChild(title)}});add(svg,'line',{{x1:x(totalMs),y1:m.t-8,x2:x(totalMs),y2:H-m.b,stroke:'#fff','stroke-width':1.2}});target.replaceChildren(svg)}}
function sourceTimeline(target,d,totalMs){{
  const W=1480,rowH=44,m={{l:58,r:22,t:28,b:38}},stageCount=d.stage_count,H=m.t+stageCount*rowH+m.b,x=v=>m.l+(W-m.l-m.r)*v/totalMs;
  const svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox',`0 0 ${{W}} ${{H}}`);
  for(let s=0;s<stageCount;s++){{const y=m.t+s*rowH;add(svg,'rect',{{x:m.l,y,width:W-m.l-m.r,height:rowH,fill:s%2?'#0a1827':'#0c1b2c'}});add(svg,'text',{{x:m.l-8,y:y+25,'text-anchor':'end',fill:'#93a8bd','font-size':11}},`PP${{s}}`)}}
  for(let t=0;t<=totalMs;t+=5000){{const xx=x(t);add(svg,'line',{{x1:xx,y1:m.t-7,x2:xx,y2:H-m.b,stroke:'#294057','stroke-dasharray':'3 4'}});add(svg,'text',{{x:xx,y:H-13,'text-anchor':'middle',fill:'#93a8bd','font-size':10}},`${{fmt(t/1000,0)}}s`)}}
  d.bars.forEach(item=>{{const track=PHASE_TRACKS[item.phase],yy=m.t+item.stage*rowH+track.y_offset,w=Math.max(1.5,x(item.end_ms)-x(item.start_ms));const r=add(svg,'rect',{{x:x(item.start_ms),y:yy,width:w,height:track.height,rx:2,fill:PHASE_COLORS[item.phase],'data-phase':item.phase,'data-source':'source256'}});const title=document.createElementNS(ns,'title');title.textContent=`256 source · PP${{item.stage}} ${{item.phase==='forward'?'F':'B'}}${{item.microbatch}} · 16-rank envelope · ${{fmt(item.start_ms/1000,3)}}–${{fmt(item.end_ms/1000,3)}}s · wall ${{fmt(item.duration_ms,2)}}ms · last rank starts ${{fmt(item.last_start_ms/1000,3)}}s · active union ${{fmt(item.active_union_ms,2)}}ms`;r.appendChild(title);if(w>27)add(svg,'text',{{x:x(item.start_ms)+3,y:yy+8,fill:'#07111d','font-size':7,'font-weight':800}},`${{item.phase==='forward'?'F':'B'}}${{item.microbatch}}`)}});
  d.collectives.forEach(item=>{{const isDp=item.group_type==='dp',isRs=item.collective==='rs',yy=m.t+item.stage*rowH+(isDp?31:37),h=5,w=Math.max(1.4,x(item.end_ms)-x(item.start_ms)),fill=isDp?(isRs?'#27d17f':'#ffb54a'):(isRs?'#00b8a9':'#b78cff');const r=add(svg,'rect',{{x:x(item.start_ms),y:yy,width:w,height:h,rx:1,fill}});const splitDpRs=isDp&&isRs&&Number.isFinite(item.last_release_ms);let earlyRect=null;if(splitDpRs){{const earlyEnd=Math.min(item.end_ms,Math.max(item.start_ms,item.last_release_ms));if(earlyEnd>item.start_ms)earlyRect=add(svg,'rect',{{x:x(item.start_ms),y:yy,width:Math.max(1.4,x(earlyEnd)-x(item.start_ms)),height:h,rx:1,fill:'#64748b'}})}}const title=document.createElementNS(ns,'title');title.textContent=(splitDpRs?'256 source DP group envelope':`256 source rank${{item.rank}} GPU kernel`)+` · PP${{item.stage}} ${{isDp?'DP':'EDP'}} ${{item.collective.toUpperCase()}} · ${{fmt(item.start_ms/1000,3)}}–${{fmt(item.end_ms/1000,3)}}s · ${{fmt(item.duration_ms,2)}}ms`+(splitDpRs?` · arrival ${{fmt(item.group_arrival_span_ms,2)}}ms · service ${{fmt(item.group_tail_after_last_release_ms,2)}}ms · rank${{item.rank}} kernel ${{fmt(item.rank_duration_ms,2)}}ms`:``);r.appendChild(title);if(earlyRect){{const earlyTitle=document.createElementNS(ns,'title');earlyTitle.textContent=`DP RS group arrival ${{fmt(item.group_arrival_span_ms,2)}}ms：最早rank到最晚rank；期间可能已有部分通信`;earlyRect.appendChild(earlyTitle)}}}});
  add(svg,'line',{{x1:x(totalMs),y1:m.t-8,x2:x(totalMs),y2:H-m.b,stroke:'#fff','stroke-width':1.2}});target.replaceChildren(svg)
}}
function ribbon(target,d){{const phase=100*d.phase_envelope_ms/d.profiler_ms,out=100-phase;target.innerHTML=`<div class="ribbon"><span class="phase" style="width:${{phase}}%">F/B envelope ${{fmt(d.phase_envelope_ms/1000,3)}}s</span><span class="outside" style="width:${{out}}%">outside ${{fmt(d.outside_phase_envelope_ms/1000,3)}}s</span></div>`}}
function ratios(rows){{$('ratioChart').innerHTML=rows.map(r=>{{const h=Math.min(100,r.trace_over_predicted/2*100);return `<div class="ratio-col"><div class="ratio-v">${{fmt(r.trace_over_predicted,2)}}×</div><div class="ratio-bar-wrap"><div class="ratio-one"></div><div class="ratio-bar ${{r.trace_over_predicted>1.2?'hot':''}}" style="height:${{h}}%"></div></div><div class="ratio-stage">PP${{r.stage}}</div></div>`}}).join('')}}
function gapTable(){{const p=DATA.adjacent_phase_gaps.predicted,t=DATA.adjacent_phase_gaps.trace[selected];$('gapTable').innerHTML='<thead><tr><th>stage</th><th>F0→F1 DAG / Trace</th><th>F1→F2 DAG / Trace</th><th>B0→B1 DAG / Trace</th><th>B1→B2 DAG / Trace</th></tr></thead><tbody>'+p.map((r,i)=>`<tr><td>PP${{r.stage}}</td><td>${{fmt(r.f01_ms,1)}} / ${{fmt(t[i].f01_ms,1)}}</td><td>${{fmt(r.f12_ms,1)}} / ${{fmt(t[i].f12_ms,1)}}</td><td>${{fmt(r.b01_ms,1)}} / ${{fmt(t[i].b01_ms,1)}}</td><td>${{fmt(r.b12_ms,1)}} / ${{fmt(t[i].b12_ms,1)}}</td></tr>`).join('')+'</tbody>'}}
function median(values){{const a=values.filter(Number.isFinite).sort((x,y)=>x-y),n=a.length;if(!n)return NaN;return n%2?a[(n-1)/2]:(a[n/2-1]+a[n/2])/2}}
function dpRsSummary(items){{const rows=items.filter(d=>d.group_type==='dp'&&d.collective==='rs'&&Number.isFinite(d.group_arrival_span_ms));return{{stages:rows.length,groupFct:median(rows.map(d=>d.duration_ms)),arrival:median(rows.map(d=>d.group_arrival_span_ms)),service:median(rows.map(d=>d.group_tail_after_last_release_ms)),rankKernel:median(rows.map(d=>d.rank_duration_ms)),rankEarly:median(rows.map(d=>d.early_residence_ms))}}}}
function dpRsTable(targetItems,sourceItems){{const rows=[['B · 224卡Trace',dpRsSummary(targetItems)],['C · 256卡源Trace',dpRsSummary(sourceItems)]];$('dpRsTable').innerHTML=`<thead><tr><th>数据</th><th>PP数</th><th>group FCT</th><th>arrival</th><th>service</th><th>lane${{DATA.representative_lane}} kernel</th><th>该lane早到驻留</th></tr></thead><tbody>`+rows.map(([name,d])=>`<tr><td>${{name}}</td><td>${{d.stages}}</td><td>${{fmt(d.groupFct,1)}} ms</td><td>${{fmt(d.arrival,1)}} ms</td><td>${{fmt(d.service,1)}} ms</td><td>${{fmt(d.rankKernel,1)}} ms</td><td>${{fmt(d.rankEarly,1)}} ms</td></tr>`).join('')+'</tbody>'}}
function render(){{
  const p=DATA.predicted.decomposition,pc=DATA.predicted.collective_span,t=DATA.trace[selected],td=t.decomposition,tc=t.collective_span,s=DATA.source_256_trace[selected];
  const max=Math.ceil(Math.max(p.profiler_ms,td.profiler_ms,s.timeline_end_ms)/5000)*5000;
  const visiblePredictedCollectives=DATA.predicted.collectives.filter(item=>item.group_type!=='handoff');
  timeline($('predictedTimeline'),DATA.predicted.bars,visiblePredictedCollectives,max,true,false);
  timeline($('traceTimeline'),t.bars,t.collectives,max,false,true);
  sourceTimeline($('sourceTimeline'),s,max);
  ribbon($('predictedRibbon'),p);ribbon($('traceRibbon'),td);
  $('predProfiler').textContent=fmt(p.profiler_ms/1000,3)+' s';$('traceProfiler').textContent=fmt(td.profiler_ms/1000,3)+' s';$('profGap').textContent=fmt((td.profiler_ms-p.profiler_ms)/1000,3)+' s';$('phaseGap').textContent=fmt((td.phase_envelope_ms-p.phase_envelope_ms)/1000,3)+' s';$('outsideGap').textContent=fmt((td.outside_phase_envelope_ms-p.outside_phase_envelope_ms)/1000,3)+' s';
  $('traceBadge').textContent='224 Trace iter '+selected+' · DP/EDP 84 nodes';
  $('sourceBadge').textContent='256 Source iter '+selected+' · DP/EDP '+s.observed_collective_count+'/'+s.expected_collective_count+(s.missing_collective_count?' · 缺'+s.missing_collective_count:'');
  ratios(DATA.stage_ratios[selected]);gapTable();dpRsTable(t.collectives,s.collectives);
  $('differenceTable').innerHTML=`<thead><tr><th>部分</th><th>DAG</th><th>224 Trace</th><th>差值</th></tr></thead><tbody><tr><td>F/B phase envelope</td><td>${{fmt(p.phase_envelope_ms,1)}} ms</td><td>${{fmt(td.phase_envelope_ms,1)}} ms</td><td>${{fmt(td.phase_envelope_ms-p.phase_envelope_ms,1)}} ms</td></tr><tr><td>DP/EDP 首次开始</td><td>${{fmt(pc.first_start_ms,1)}} ms</td><td>${{fmt(tc.first_start_ms,1)}} ms</td><td>${{fmt(tc.first_start_ms-pc.first_start_ms,1)}} ms</td></tr><tr><td>DP/EDP 最后完成</td><td>${{fmt(pc.last_end_ms,1)}} ms</td><td>${{fmt(tc.last_end_ms,1)}} ms</td><td>${{fmt(tc.last_end_ms-pc.last_end_ms,1)}} ms</td></tr><tr><td>DP/EDP 可见跨度</td><td>${{fmt(pc.span_ms,1)}} ms</td><td>${{fmt(tc.span_ms,1)}} ms</td><td>${{fmt(tc.span_ms-pc.span_ms,1)}} ms</td></tr><tr><td>Profiler 中F/B包络外</td><td>${{fmt(p.outside_phase_envelope_ms,1)}} ms</td><td>${{fmt(td.outside_phase_envelope_ms,1)}} ms</td><td>${{fmt(td.outside_phase_envelope_ms-p.outside_phase_envelope_ms,1)}} ms</td></tr><tr><td>Profiler Step</td><td>${{fmt(p.profiler_ms,1)}} ms</td><td>${{fmt(td.profiler_ms,1)}} ms</td><td>${{fmt(td.profiler_ms-p.profiler_ms,1)}} ms</td></tr></tbody>`;
  document.querySelectorAll('#iterationControls button').forEach(b=>b.classList.toggle('active',b.dataset.it===selected));document.querySelectorAll('#dependencyControls button').forEach(b=>b.classList.toggle('active',b.dataset.dep===depMode))
}}
function harmonizeTimelineStyles(){{const svg=$('predictedTimeline').querySelector('svg');if(!svg)return;[...svg.querySelectorAll('rect')].forEach(rect=>{{const title=rect.querySelector('title'),label=title?.textContent||'';if(/^PP\\d+ [FB]\\d+/.test(label)){{const critical=rect.getAttribute('stroke')==='#ffe45e';rect.setAttribute('stroke','none');rect.setAttribute('stroke-width','0');if(critical){{const x=Number(rect.getAttribute('x')),y=Number(rect.getAttribute('y')),w=Number(rect.getAttribute('width'));add(svg,'line',{{x1:x,y1:y-1,x2:x+w,y2:y-1,stroke:'#ffe45e','stroke-width':2}})}}}}if(label.includes('AG_ROUND')){{rect.setAttribute('fill','#f4d35e');rect.setAttribute('stroke','none');if(title)title.textContent=label.replace('EDP AG_ROUND','AG ROUND BARRIER')}}if(label.includes(' F2B ')||label.includes(' B2F ')){{rect.setAttribute('fill','#ff66c4');rect.setAttribute('stroke','none');if(title)title.textContent=label.replace('EDP F2B','PHASE HANDOFF F→B').replace('EDP B2F','PHASE HANDOFF B→F')}}}})}}
function splitPredictedDpRs(){{
  const svg=$('predictedTimeline').querySelector('svg');if(!svg)return;
  const trace=DATA.trace[selected],source=DATA.source_256_trace[selected],totalMs=Math.ceil(Math.max(DATA.predicted.decomposition.profiler_ms,trace.decomposition.profiler_ms,source.timeline_end_ms)/5000)*5000;
  const x=value=>58+(1480-58-22)*value/totalMs,rects=[...svg.querySelectorAll('rect')];
  for(const item of DATA.predicted.collectives.filter(d=>d.group_type==='dp'&&d.collective==='rs'&&Number.isFinite(d.last_release_ms))){{
    const rect=rects.find(candidate=>(candidate.querySelector('title')?.textContent||'').startsWith(`DAG group service · PP${{item.stage}} DP RS ·`));
    if(!rect)continue;
    const earlyEnd=Math.min(item.end_ms,Math.max(item.start_ms,item.last_release_ms));if(earlyEnd<=item.start_ms)continue;
    const early=add(svg,'rect',{{x:x(item.start_ms),y:rect.getAttribute('y'),width:Math.max(1.4,x(earlyEnd)-x(item.start_ms)),height:rect.getAttribute('height'),rx:1,fill:'#64748b'}});
    const title=document.createElementNS(ns,'title');title.textContent=`DAG DP RS 到达/同步跨度 ${{fmt(item.group_arrival_span_ms,2)}}ms · 由${{item.group_observed_rank_count}}个rank就绪节点形成，已进入DAG调度，不重复串行计费`;early.appendChild(title);
  }}
}}
function dpRsTableABC(predictedItems,targetItems,sourceItems){{
  const rows=[['A · v67预测',dpRsSummary(predictedItems)],['B · 224卡Trace',dpRsSummary(targetItems)],['C · 256卡源Trace',dpRsSummary(sourceItems)]],cell=value=>Number.isFinite(value)?`${{fmt(value,1)}} ms`:'—';
  $('dpRsTable').innerHTML='<thead><tr><th>数据</th><th>PP数</th><th>group FCT</th><th>arrival/同步</th><th>service</th><th>代表lane kernel</th><th>代表lane早到驻留</th></tr></thead><tbody>'+rows.map(([name,d])=>`<tr><td>${{name}}</td><td>${{d.stages}}</td><td>${{cell(d.groupFct)}}</td><td>${{cell(d.arrival)}}</td><td>${{cell(d.service)}}</td><td>${{cell(d.rankKernel)}}</td><td>${{cell(d.rankEarly)}}</td></tr>`).join('')+'</tbody>';
}}
function applySharedPhaseStyles(){{for(const target of ['predictedTimeline','traceTimeline']){{const svg=$(target).querySelector('svg');if(!svg)continue;for(const rect of svg.querySelectorAll('rect')){{const label=rect.querySelector('title')?.textContent||'',match=label.match(/^PP(\\d+) ([FB])\\d+/);if(!match)continue;const phase=match[2]==='F'?'forward':'backward',track=PHASE_TRACKS[phase];rect.setAttribute('y',28+Number(match[1])*44+track.y_offset);rect.setAttribute('height',track.height);rect.setAttribute('fill',PHASE_COLORS[phase]);rect.setAttribute('data-phase',phase);rect.setAttribute('data-source',target==='predictedTimeline'?'prediction':'trace')}}}}}}
function renderStyled(){{render();applySharedPhaseStyles();harmonizeTimelineStyles();splitPredictedDpRs();dpRsTableABC(DATA.predicted.collectives,DATA.trace[selected].collectives,DATA.source_256_trace[selected].collectives)}}
$('iterationControls').innerHTML=DATA.iterations.map(it=>`<button data-it="${{it}}">iter ${{it}}</button>`).join('');$('iterationControls').addEventListener('click',e=>{{if(e.target.dataset.it){{selected=e.target.dataset.it;renderStyled()}}}});$('dependencyControls').addEventListener('click',e=>{{if(e.target.dataset.dep){{depMode=e.target.dataset.dep;renderStyled()}}}});renderStyled();</script></main></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--trace-metadata", type=Path, default=DEFAULT_TRACE_METADATA)
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--collective-events", type=Path, default=DEFAULT_COLLECTIVE_EVENTS)
    parser.add_argument("--model-version", choices=("v62", "v63", "v631", "v64a", "v65", "v66", "v67"), default=None)
    parser.add_argument(
        "--representative-lane", type=int, default=None,
        help="Use a trace-complete PP lane explicitly; does not change model prediction.",
    )
    args = parser.parse_args()
    run = args.run_dir.resolve()
    model_version = args.model_version or (
        "v67" if "dag_v67" in run.name else "v66" if "dag_v66" in run.name else "v65" if "dag_v65" in run.name else "v64a" if "dag_v64a" in run.name
        else "v631" if "dag_v631" in run.name else "v63" if "dag_v63" in run.name else "v62"
    )
    payload, inputs = build_payload(
        run,
        model_version,
        args.manifest.resolve(),
        args.trace_root.resolve(),
        args.collective_events.resolve(),
        args.trace_metadata.resolve(),
        args.representative_lane,
    )
    evaluator = run / "evaluator_only"
    payload_path = evaluator / f"dag_{model_version}_trace_timeline_payload.json"
    html_path = evaluator / f"dag_{model_version}_predicted_vs_trace_global_timeline.html"
    evolution_path = evaluator / f"dag_{model_version}_html_evolution_log.md"
    data_scope_path = evaluator / f"dag_{model_version}_224_data_scope_audit.md"
    provenance_path = evaluator / f"dag_{model_version}_trace_timeline_provenance.json"
    atomic_json(payload_path, payload)
    atomic_text(html_path, render_html(payload))
    atomic_text(
        evolution_path,
        "# 全局时序 HTML 演进日志\n\n"
        + "\n".join(
            f"- `{item['version']}`：{item['change']}"
            for item in payload["html_evolution"]
        )
        + f"\n\n当前 FWD/BWD 配色分别为 `{PHASE_COLORS['forward']}` / "
        + f"`{PHASE_COLORS['backward']}`，上下轨偏移分别为 "
        + f"`{PHASE_TRACKS['forward']['y_offset']} px` / "
        + f"`{PHASE_TRACKS['backward']['y_offset']} px`；预测和 Trace 共用。\n",
    )
    scope = payload["data_scope_audit"]
    atomic_text(
        data_scope_path,
        f"# DAG {display_version(model_version)} 的 224 卡数据边界审计\n\n"
        f"- 状态：`{scope['status']}`。\n"
        f"- 预测进程直接读取的 224 卡时序文件：`{scope['model_process_target_timing_files_read']}`。\n"
        f"- 预测时序来源：`{scope['model_source_trace_path']}`，iterations "
        f"`{scope['model_source_iterations']}`。\n"
        f"- 224 卡 Trace 的角色：`{scope['target_trace_role']}`；只在 prediction seal 后由 evaluator 读取。\n"
        f"- F/B 可视化粒度：`{scope['phase_bar_granularity']}`。A/B/C 都以最早 rank 开始到最晚 rank 完成为一个 bar。\n\n"
        "## 结论\n\n"
        "代码层面没有把 224 卡时序直接喂给 v6.7 预测进程；但 v6 系列方法开发已经看过 224 卡诊断，"
        "因此当前 224 卡数字是回顾性开发集结果，不是未见目标数据的盲测精度，也不能据此宣称正式外推胜负。\n",
    )
    atomic_json(provenance_path, {
        "schema": f"dag-{display_version(model_version)}-trace-timeline-provenance-v1",
        "status": "PASS_EVALUATOR_ONLY_NO_MODEL_UPDATE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "representative_lane": payload["representative_lane"],
        "target_trace_files": 224 * len(ITERATIONS),
        "target_phase_group_rows": 14 * 2 * 3 * len(ITERATIONS),
        "target_trace_collective_rows": 14 * 6 * len(ITERATIONS),
        "target_trace_iterations": list(ITERATIONS),
        "model_parameter_updates": 0,
        "builder": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "inputs": [
            {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in inputs
        ],
        "outputs": [
            {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (payload_path, html_path, evolution_path, data_scope_path)
        ],
    })
    manifest_path = evaluator / "artifact_manifest.json"
    artifacts = sorted(
        path.resolve() for path in evaluator.rglob("*")
        if path.is_file() and path.resolve() != manifest_path.resolve() and not path.name.endswith(".tmp")
    )
    atomic_json(manifest_path, {
        "schema": f"dag-{display_version(model_version)}-evaluation-artifact-manifest-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in artifacts
        ],
    })
    print(json.dumps({
        "status": "PASS_EVALUATOR_ONLY_NO_MODEL_UPDATE", "representative_lane": payload["representative_lane"],
        "html": str(html_path), "trace_iterations": list(ITERATIONS),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
