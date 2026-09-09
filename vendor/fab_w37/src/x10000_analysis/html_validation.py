from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import struct
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path, PurePosixPath
from typing import Any, Callable, TextIO

from bs4 import BeautifulSoup, NavigableString, Tag


_UNRESOLVED_TEMPLATE = re.compile(r"{{[^{}]+}}")
_REMOTE_ADDRESS = re.compile(r"^(?:https?:)?//", re.IGNORECASE)
_REMOTE_CSS_IMPORT = re.compile(r"@import\b", re.IGNORECASE)
_REMOTE_CSS_URL = re.compile(
    r"url\(\s*(['\"]?)\s*(?:https?:)?//", re.IGNORECASE
)
_VISIBLE_PLACEHOLDER = re.compile(r"\b(?:TODO|TBD|placeholder)\b|占位", re.IGNORECASE)
_ARIA_IDREF_ATTRIBUTES = ("aria-controls", "aria-labelledby", "aria-describedby")


@dataclass(frozen=True, order=True)
class Failure:
    code: str
    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _failure(code: str, path: str, message: str) -> Failure:
    return Failure(code=code, path=path, message=message)


def _sorted(failures: list[Failure]) -> list[Failure]:
    return sorted(failures, key=lambda item: (item.code, item.path, item.message))


def strict_json_loads(text: str) -> Any:
    """Parse strict JSON and reject JavaScript-only non-finite constants."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    try:
        return json.loads(text, parse_constant=reject_constant)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"invalid strict JSON: {exc}") from exc


def validate_size_bytes(size: int, *, limit: int, path: str) -> list[Failure]:
    """Apply a strict '< limit' artifact byte budget."""
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        return [_failure("ARTIFACT_SIZE_INVALID", path, "artifact byte size must be a non-negative integer")]
    if size >= limit:
        return [
            _failure(
                "ARTIFACT_SIZE_LIMIT",
                path,
                f"artifact must be strictly smaller than {limit} bytes; got {size}",
            )
        ]
    return []


def _is_visible(element: Tag) -> bool:
    current: Tag | None = element
    while current is not None:
        if current.name in {"head", "script", "style", "template", "noscript"}:
            return False
        if current.has_attr("hidden") or str(current.get("aria-hidden", "")).lower() == "true":
            return False
        style = re.sub(r"\s+", "", str(current.get("style", "")).lower())
        if "display:none" in style or "visibility:hidden" in style:
            return False
        parent = current.parent
        current = parent if isinstance(parent, Tag) else None
    return True


def _visible_text(soup: BeautifulSoup) -> str:
    parts: list[str] = []
    for node in soup.find_all(string=True):
        if isinstance(node, NavigableString) and isinstance(node.parent, Tag) and _is_visible(node.parent):
            parts.append(str(node))
    return " ".join(parts)


def _has_label(control: Tag, soup: BeautifulSoup, id_counts: dict[str, int]) -> bool:
    if control.get("aria-label"):
        return True
    labelledby = str(control.get("aria-labelledby", "")).split()
    if labelledby and all(id_counts.get(token) == 1 for token in labelledby):
        return True
    if control.find_parent("label") is not None:
        return True
    control_id = control.get("id")
    return bool(control_id and soup.find("label", attrs={"for": control_id}) is not None)


def validate_html_document(text: str, *, path: str, product: str = "generic") -> list[Failure]:
    """Validate one self-contained final HTML document without reading or writing files."""
    failures: list[Failure] = []
    soup = BeautifulSoup(text, "html.parser")

    html = soup.html
    if html is None or html.get("lang") != "zh-CN":
        failures.append(_failure("HTML_LANG", path, 'html lang must equal "zh-CN"'))
    charset = soup.find("meta", attrs={"charset": True})
    if charset is None or str(charset.get("charset", "")).lower().replace("_", "-") != "utf-8":
        failures.append(_failure("HTML_CHARSET", path, "UTF-8 meta charset is required"))
    viewport = soup.find("meta", attrs={"name": lambda value: str(value).lower() == "viewport"})
    if viewport is None or not str(viewport.get("content", "")).strip():
        failures.append(_failure("HTML_VIEWPORT", path, "non-empty viewport metadata is required"))
    if soup.title is None or not soup.title.get_text(" ", strip=True):
        failures.append(_failure("HTML_TITLE", path, "non-empty title is required"))

    payload_tags = soup.select('script#run1555-payload[type="application/json"]')
    if len(payload_tags) != 1:
        failures.append(
            _failure(
                "HTML_PAYLOAD_COUNT",
                path,
                "exactly one application/json script#run1555-payload is required",
            )
        )
    else:
        try:
            strict_json_loads(payload_tags[0].string or payload_tags[0].get_text())
        except ValueError as exc:
            failures.append(_failure("HTML_PAYLOAD_JSON", path, str(exc)))

    for script in soup.select("script[src]"):
        failures.append(_failure("HTML_SCRIPT_SRC", path, "final HTML must not contain script[src]"))
    for link in soup.select("link[rel]"):
        rel = {str(value).lower() for value in link.get("rel", [])}
        if "stylesheet" in rel:
            failures.append(
                _failure("HTML_STYLESHEET_LINK", path, "final HTML must not contain linked stylesheets")
            )
    for element in soup.select("[src], [href]"):
        for attribute in ("src", "href"):
            if not element.has_attr(attribute):
                continue
            address = str(element.get(attribute, "")).strip()
            if _REMOTE_ADDRESS.match(address):
                failures.append(
                    _failure(
                        "HTML_REMOTE_REFERENCE",
                        path,
                        f"remote {attribute} is forbidden: {address}",
                    )
                )
    for style in soup.find_all("style"):
        css = style.string or style.get_text()
        if _REMOTE_CSS_IMPORT.search(css):
            failures.append(_failure("HTML_CSS_IMPORT", path, "owned CSS must not contain @import"))
        if _REMOTE_CSS_URL.search(css):
            failures.append(_failure("HTML_CSS_REMOTE_URL", path, "owned CSS must not contain remote url()"))

    if _UNRESOLVED_TEMPLATE.search(text):
        failures.append(
            _failure(
                "HTML_UNRESOLVED_TEMPLATE",
                path,
                "unresolved template expression matching {{[^{}]+}} is forbidden",
            )
        )

    visible_placeholder_elements = [
        element
        for element in soup.select('[data-placeholder="true"]')
        if _is_visible(element)
    ]
    if visible_placeholder_elements or _VISIBLE_PLACEHOLDER.search(_visible_text(soup)):
        failures.append(
            _failure("HTML_VISIBLE_PLACEHOLDER", path, "visible placeholder/TODO/TBD content is forbidden")
        )

    id_counts: dict[str, int] = {}
    for element in soup.select("[id]"):
        element_id = str(element.get("id"))
        id_counts[element_id] = id_counts.get(element_id, 0) + 1
    duplicates = sorted(element_id for element_id, count in id_counts.items() if count != 1)
    if duplicates:
        failures.append(
            _failure("HTML_DUPLICATE_ID", path, f"IDs must be unique; duplicates={duplicates}")
        )

    broken_idrefs: list[str] = []
    for element in soup.find_all(True):
        for attribute in _ARIA_IDREF_ATTRIBUTES:
            for token in str(element.get(attribute, "")).split():
                if id_counts.get(token) != 1:
                    broken_idrefs.append(f"{attribute}={token}")
    if broken_idrefs:
        failures.append(
            _failure(
                "HTML_ARIA_IDREF",
                path,
                f"ARIA references must resolve to unique IDs; broken={sorted(set(broken_idrefs))}",
            )
        )

    unlabeled = [
        control.name
        for control in soup.select("input, select, textarea")
        if not _has_label(control, soup, id_counts)
    ]
    if unlabeled:
        failures.append(
            _failure("HTML_FORM_LABEL", path, f"all form controls require labels; unlabeled={unlabeled}")
        )

    bad_buttons = [button.get("id", "<no-id>") for button in soup.find_all("button") if button.get("type") != "button"]
    if bad_buttons:
        failures.append(
            _failure(
                "HTML_BUTTON_TYPE",
                path,
                f'all buttons must explicitly use type="button"; invalid={bad_buttons}',
            )
        )

    if product == "dashboard":
        tabs = soup.select('[role="tab"]')
        panels = soup.select('[role="tabpanel"]')
        relation_ok = len(tabs) == len(panels) == 4
        if relation_ok:
            panel_ids = {str(panel.get("id")) for panel in panels}
            tab_ids = {str(tab.get("id")) for tab in tabs}
            relation_ok = all(
                tab.get("id")
                and tab.get("aria-controls") in panel_ids
                and id_counts.get(str(tab.get("aria-controls"))) == 1
                for tab in tabs
            ) and all(
                panel.get("id")
                and panel.get("aria-labelledby") in tab_ids
                and id_counts.get(str(panel.get("aria-labelledby"))) == 1
                for panel in panels
            )
            relation_ok = relation_ok and {
                str(tab.get("aria-controls")) for tab in tabs
            } == panel_ids and {str(panel.get("aria-labelledby")) for panel in panels} == tab_ids
        if not relation_ok:
            failures.append(
                _failure(
                    "DASHBOARD_TAB_RELATION",
                    path,
                    "Dashboard requires four bidirectionally closed tab/tabpanel relationships",
                )
            )

    return _sorted(failures)


DIRECT_PROFILER_ITERS = [
    4,
    8,
    12,
    16,
    20,
    24,
    28,
    32,
    36,
    40,
    44,
    48,
    52,
    56,
    60,
    64,
    68,
    72,
    76,
    80,
    84,
    88,
    92,
    96,
]

_DASHBOARD_ROOT_KEYS = {
    "schema_version",
    "metadata",
    "validator",
    "run",
    "stage_taxonomy",
    "semantics",
    "profiler_view",
    "evidence",
}
_GUIDE_ROOT_KEYS = {
    "schema_version",
    "scope",
    "topology",
    "groups",
    "parallelism",
    "batch",
    "config_facts",
    "objects",
    "steps",
    "primitives",
    "concepts",
    "cases",
    "metrics",
    "boundaries",
    "claim_audit",
    "quiz",
}
_GUIDE_OBJECT_NAMES = ["Token", "Activation", "Gradient", "Optimizer shard", "Parameter"]


def _rows(container: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    value = container.get(key)
    if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
        raise ValueError(f"{key} must be an array of objects")
    return value


def _iteration_values(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        result: list[int] = []
        if "iter" in value:
            iteration = value["iter"]
            if isinstance(iteration, bool) or not isinstance(iteration, int):
                raise ValueError(f"iter must be an integer; got {iteration!r}")
            result.append(iteration)
        for nested in value.values():
            result.extend(_iteration_values(nested))
        return result
    if isinstance(value, list):
        return [item for nested in value for item in _iteration_values(nested)]
    return []


def _phase_interval_count(groups: Sequence[Mapping[str, Any]]) -> int:
    count = 0
    for group in groups:
        intervals = group.get("intervals")
        if not isinstance(intervals, list):
            raise ValueError("phase_intervals group must contain an intervals array")
        count += len(intervals)
    return count


def _dense_grid(
    rows: Sequence[Mapping[str, Any]],
    iterations: Sequence[int],
    entity_fields: Sequence[str],
    *,
    expected_entities: set[tuple[Any, ...]] | None = None,
) -> tuple[bool, set[tuple[Any, ...]]]:
    keys: list[tuple[int, tuple[Any, ...]]] = []
    for row in rows:
        if "iter" not in row or any(field not in row for field in entity_fields):
            return False, set()
        iteration = row["iter"]
        if isinstance(iteration, bool) or not isinstance(iteration, int):
            return False, set()
        entity = tuple(row[field] for field in entity_fields)
        keys.append((iteration, entity))
    entities = {entity for _, entity in keys}
    if expected_entities is not None and entities != expected_entities:
        return False, entities
    expected = set(product(iterations, entities if expected_entities is None else expected_entities))
    return Counter(keys) == Counter({key: 1 for key in expected}), entities


def _summary_closes(
    rows: Sequence[Mapping[str, Any]],
    entity_fields: Sequence[str],
    expected_entities: set[tuple[Any, ...]],
    *,
    n_iter: int = 24,
) -> bool:
    try:
        keys = [tuple(row[field] for field in entity_fields) for row in rows]
    except KeyError:
        return False
    return (
        len(keys) == len(expected_entities)
        and len(set(keys)) == len(keys)
        and set(keys) == expected_entities
        and all(row.get("n_iter") == n_iter for row in rows)
    )


def _exact_iteration_order(rows: Sequence[Mapping[str, Any]], expected: Sequence[int]) -> bool:
    try:
        return [row["iter"] for row in rows] == list(expected)
    except KeyError:
        return False


def validate_canonical_payload(
    payload: Mapping[str, Any], *, path: str = "derived_v5/run1555_html_payload_v5.json"
) -> list[Failure]:
    """Validate full-99 canonical audit counts separately from the strict browser view."""
    failures: list[Failure] = []
    if not isinstance(payload, Mapping):
        return [_failure("CANONICAL_STRUCTURE", path, "canonical payload root must be an object")]
    if payload.get("schema_version") != "run1555-html-payload-v5.2":
        failures.append(
            _failure("CANONICAL_SCHEMA", path, 'schema_version must equal "run1555-html-payload-v5.2"')
        )
    validator = payload.get("validator")
    statuses_ok = isinstance(validator, Mapping) and all(
        isinstance(validator.get(key), Mapping) and validator[key].get("status") == "PASS"
        for key in ("m1", "full_v5")
    )
    statuses_ok = statuses_ok and validator.get("status") == "PASS" if isinstance(validator, Mapping) else False
    if not statuses_ok:
        failures.append(
            _failure("CANONICAL_VALIDATOR_STATUS", path, "validator, m1, and full_v5 statuses must all be PASS")
        )
    metadata = payload.get("metadata")
    inputs = metadata.get("inputs") if isinstance(metadata, Mapping) else None
    if not isinstance(inputs, list) or len(inputs) != 21:
        failures.append(
            _failure("CANONICAL_INPUT_COUNT", path, "metadata.inputs must contain exactly 21 records")
        )
    expected_coverage = {
        "iter_start": 1,
        "iter_end": 99,
        "iter_count": 99,
        "excluded_iters": [100],
        "direct_profiler_iters": DIRECT_PROFILER_ITERS,
        "direct_profiler_count": 24,
        "validated_inference_count": 75,
    }
    run = payload.get("run")
    if not isinstance(run, Mapping) or run.get("coverage") != expected_coverage:
        failures.append(
            _failure(
                "CANONICAL_COVERAGE",
                path,
                "run.coverage must exactly state audit iter1..99, excluded iter100, explicit24, and inferred75",
            )
        )

    try:
        data = payload["data"]
        if not isinstance(data, Mapping):
            raise ValueError("data must be an object")
        iter_summary = _rows(data, "iter_summary")
        stage_share = _rows(data, "stage_share")
        phase_groups = _rows(data, "phase_intervals")
        headroom = _rows(data, "stage_headroom")
        joint = _rows(data, "joint_counterfactual")
        calls = _rows(data, "collective_calls")
        spatial = data["spatial_temporal"]
        if not isinstance(spatial, Mapping):
            raise ValueError("spatial_temporal must be an object")

        counts_ok = (
            len(iter_summary) == 99
            and len(stage_share) == 1188
            and len(phase_groups) == 99
            and _phase_interval_count(phase_groups) == 35242
            and len(headroom) == 891
            and len(joint) == 24
            and len(calls) == 840
            and len(_rows(spatial["nic"], "per_iter_device")) == 792
            and len(_rows(spatial["mtlink"], "per_iter_gpu")) == 1584
            and len(_rows(spatial["deepep"], "iter_summary")) == 198
        )
        if not counts_ok:
            failures.append(
                _failure(
                    "CANONICAL_AUDIT_COUNTS",
                    path,
                    "full audit counts must be iter99/stage1188/phase99+35242/headroom891/joint24/calls840/NIC792/MTLink1584/DeepEP198",
                )
            )

        full_iterations = list(range(1, 100))
        stage_ids = {
            row.get("id") for row in payload.get("stage_taxonomy", []) if isinstance(row, Mapping)
        }
        stage_grid_ok, _ = _dense_grid(
            stage_share, full_iterations, ["stage"], expected_entities={(stage,) for stage in stage_ids}
        )
        headroom_stages = {row.get("stage") for row in headroom}
        headroom_ok, _ = _dense_grid(
            headroom,
            full_iterations,
            ["stage"],
            expected_entities={(stage,) for stage in headroom_stages},
        )
        audit_domain_ok = (
            _exact_iteration_order(iter_summary, full_iterations)
            and _exact_iteration_order(phase_groups, full_iterations)
            and stage_grid_ok
            and len(stage_ids) == 12
            and headroom_ok
            and len(headroom_stages) == 9
            and _exact_iteration_order(joint, DIRECT_PROFILER_ITERS)
            and set(_iteration_values(data)) <= set(full_iterations)
            and 100 not in _iteration_values(data)
        )
        if not audit_domain_ok:
            failures.append(
                _failure(
                    "CANONICAL_AUDIT_DOMAIN",
                    path,
                    "audit rows must close over iter1..99, direct-only rows over explicit24, and exclude iter100",
                )
            )
    except (KeyError, TypeError, ValueError) as exc:
        failures.append(_failure("CANONICAL_STRUCTURE", path, f"canonical data structure is invalid: {exc}"))
    return _sorted(failures)


def _edp_stage(collective: str) -> str | None:
    if collective == "allgather_into_tensor_coalesced":
        return "edp_ag_scaleout"
    if collective == "reduce_scatter_tensor_coalesced":
        return "edp_rs_scaleout"
    return None


def validate_dashboard_projection(
    payload: Mapping[str, Any], *, path: str = "html_v5/run1555_v5_analysis_dashboard.html"
) -> list[Failure]:
    """Validate the exact strict-24 browser projection and entity-domain closure."""
    failures: list[Failure] = []
    if not isinstance(payload, Mapping):
        return [_failure("DASHBOARD_STRUCTURE", path, "Dashboard payload root must be an object")]
    if set(payload) != _DASHBOARD_ROOT_KEYS:
        failures.append(
            _failure(
                "DASHBOARD_ROOT_KEYS",
                path,
                "Dashboard root keys must be exact and must not contain root data/kpis/conclusion_anchors",
            )
        )
    if payload.get("schema_version") != "run1555-html-payload-v5.2":
        failures.append(_failure("DASHBOARD_SCHEMA", path, "Dashboard payload schema is not v5.2"))

    try:
        coverage = payload["run"]["coverage"]
        scope = payload["profiler_view"]["scope"]
        expected_scope = {
            "mode": "direct_profiler_only",
            "iterations": DIRECT_PROFILER_ITERS,
            "iter_count": 24,
            "excluded_inference_count": 75,
            "membership_source": "run.coverage.direct_profiler_iters",
        }
        if scope != expected_scope or coverage.get("direct_profiler_iters") != DIRECT_PROFILER_ITERS:
            failures.append(
                _failure(
                    "DASHBOARD_SCOPE_MISMATCH",
                    path,
                    "profiler_view.scope must equal the exact ordered explicit direct membership",
                )
            )

        data = payload["profiler_view"]["data"]
        iter_rows = _rows(data, "iter_summary")
        stage_rows = _rows(data, "stage_share")
        phase_rows = _rows(data, "phase_intervals")
        headroom_rows = _rows(data, "stage_headroom")
        joint_rows = _rows(data, "joint_counterfactual")
        calls = _rows(data, "collective_calls")
        spatial = data["spatial_temporal"]
        if not isinstance(spatial, Mapping):
            raise ValueError("profiler_view.data.spatial_temporal must be an object")
        edp = spatial["edp"]
        nic = spatial["nic"]
        mtlink = spatial["mtlink"]
        deepep = spatial["deepep"]
        if not all(isinstance(section, Mapping) for section in (edp, nic, mtlink, deepep)):
            raise ValueError("spatial source sections must be objects")

        counts_ok = (
            len(iter_rows) == 24
            and len(stage_rows) == 288
            and len(phase_rows) == 24
            and _phase_interval_count(phase_rows) == 31074
            and len(headroom_rows) == 216
            and len(joint_rows) == 24
            and len(calls) == 840
            and len(_rows(edp, "entity_summary")) == 24
            and len(_rows(edp, "direct_summary")) == 16
            and len(_rows(nic, "per_iter_device")) == 192
            and all(
                len(_rows(nic, key)) == 8
                for key in ("bytes_entity_summary", "rate_entity_summary", "active_rate_entity_summary")
            )
            and len(_rows(mtlink, "per_iter_gpu")) == 384
            and len(_rows(mtlink, "entity_summary")) == 16
            and len(_rows(deepep, "iter_summary")) == 48
            and len(_rows(deepep, "entity_summary")) == 32
        )
        if not counts_ok:
            failures.append(
                _failure(
                    "DASHBOARD_BROWSER_COUNTS",
                    path,
                    "strict browser counts must match the complete explicit-24 contract",
                )
            )

        stage_ids = {row.get("id") for row in payload["stage_taxonomy"]}
        stage_grid_ok, _ = _dense_grid(
            stage_rows,
            DIRECT_PROFILER_ITERS,
            ["stage"],
            expected_entities={(stage,) for stage in stage_ids},
        )
        headroom_stages = {row.get("stage") for row in headroom_rows}
        headroom_grid_ok, _ = _dense_grid(
            headroom_rows,
            DIRECT_PROFILER_ITERS,
            ["stage"],
            expected_entities={(stage,) for stage in headroom_stages},
        )
        call_fields = (
            "pg_name",
            "pg_description",
            "collective",
            "size_bytes",
            "occurrence_index",
            "pair",
            "rank_count",
            "algo",
            "protocol",
        )
        calls_grid_ok, call_entities = _dense_grid(calls, DIRECT_PROFILER_ITERS, call_fields)
        direct_domain_ok = (
            _exact_iteration_order(iter_rows, DIRECT_PROFILER_ITERS)
            and _exact_iteration_order(phase_rows, DIRECT_PROFILER_ITERS)
            and _exact_iteration_order(joint_rows, DIRECT_PROFILER_ITERS)
            and stage_grid_ok
            and len(stage_ids) == 12
            and headroom_grid_ok
            and len(headroom_stages) == 9
            and calls_grid_ok
            and len(call_entities) == 35
            and set(_iteration_values(payload["profiler_view"])) == set(DIRECT_PROFILER_ITERS)
        )
        if not direct_domain_ok:
            failures.append(
                _failure(
                    "DASHBOARD_DIRECT_DOMAIN",
                    path,
                    "every browser iter field and dense table must close over explicit24 only",
                )
            )

        if any(row.get("reference_count") != 24 for row in headroom_rows):
            failures.append(
                _failure(
                    "DASHBOARD_REFERENCE_COUNT",
                    path,
                    "every stage_headroom.reference_count must equal 24",
                )
            )

        closure_ok = True
        nic_atoms = _rows(nic, "per_iter_device")
        nic_grid_ok, nic_entities = _dense_grid(
            nic_atoms, DIRECT_PROFILER_ITERS, ("source", "host", "device")
        )
        closure_ok = closure_ok and nic_grid_ok and len(nic_entities) == 8
        for key in ("bytes_entity_summary", "rate_entity_summary", "active_rate_entity_summary"):
            closure_ok = closure_ok and _summary_closes(
                _rows(nic, key), ("source", "host", "device"), nic_entities
            )

        mt_atoms = _rows(mtlink, "per_iter_gpu")
        mt_grid_ok, mt_entities = _dense_grid(
            mt_atoms, DIRECT_PROFILER_ITERS, ("host", "device")
        )
        closure_ok = closure_ok and mt_grid_ok and len(mt_entities) == 16
        closure_ok = closure_ok and _summary_closes(
            _rows(mtlink, "entity_summary"), ("host", "device"), mt_entities
        )

        deep_iter = _rows(deepep, "iter_summary")
        deep_grid_ok, deep_stage_entities = _dense_grid(
            deep_iter, DIRECT_PROFILER_ITERS, ("stage_hint",)
        )
        world_size = payload["run"]["topology"]["world_size"]
        deep_entities = {
            (stage[0], rank) for stage in deep_stage_entities for rank in range(world_size)
        }
        closure_ok = closure_ok and deep_grid_ok and len(deep_stage_entities) == 2
        closure_ok = closure_ok and _summary_closes(
            _rows(deepep, "entity_summary"), ("stage_hint", "rank"), deep_entities
        )

        edp_calls = [
            row
            for row in calls
            if row.get("pg_description") == "EXPERT_DATA_PARALLEL_GROUP"
            and _edp_stage(str(row.get("collective"))) is not None
        ]
        synthetic_edp = [
            {
                "iter": row["iter"],
                "stage": _edp_stage(str(row["collective"])),
                "collective": row["collective"],
                "occurrence_index": row["occurrence_index"],
                "expected_ranks": row["pair"],
            }
            for row in edp_calls
        ]
        edp_fields = ("stage", "collective", "occurrence_index", "expected_ranks")
        edp_grid_ok, edp_entities = _dense_grid(
            synthetic_edp, DIRECT_PROFILER_ITERS, edp_fields
        )
        closure_ok = closure_ok and edp_grid_ok and len(edp_entities) == 24
        closure_ok = closure_ok and _summary_closes(
            _rows(edp, "entity_summary"), edp_fields, edp_entities
        )
        expected_direct = {(entity[3], entity[1]) for entity in edp_entities}
        direct_rows = _rows(edp, "direct_summary")
        direct_keys = [(row.get("pair"), row.get("collective")) for row in direct_rows]
        observed_counts = Counter((row["pair"], row["collective"]) for row in edp_calls)
        closure_ok = closure_ok and (
            len(direct_keys) == len(expected_direct) == 16
            and len(set(direct_keys)) == len(direct_keys)
            and set(direct_keys) == expected_direct
            and all(row.get("n") == observed_counts[(row.get("pair"), row.get("collective"))] for row in direct_rows)
        )

        if not closure_ok:
            failures.append(
                _failure(
                    "DASHBOARD_ENTITY_CLOSURE",
                    path,
                    "dense entity×explicit24 grids and exact unique summary entity keys must agree",
                )
            )

        evidence = payload["evidence"]
        expected_counts = {
            "iter_count": 99,
            "stage_share_rows": 1188,
            "stage_headroom_rows": 891,
            "nic_device_iter_rows": 792,
            "mtlink_gpu_iter_rows": 1584,
            "deepep_iter_summary_rows": 198,
        }
        if not isinstance(evidence, Mapping) or evidence.get("full_audit_counts") != expected_counts or "inference" not in evidence:
            failures.append(
                _failure(
                    "DASHBOARD_EVIDENCE_BOUNDARY",
                    path,
                    "full99 counts and inference must be retained only in evidence",
                )
            )
    except (KeyError, TypeError, ValueError) as exc:
        failures.append(
            _failure("DASHBOARD_STRUCTURE", path, f"Dashboard payload structure is invalid: {exc}")
        )
    return _sorted(failures)


def _finite_tree(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    return False


def _standalone_75_paths(value: Any, path: str = "$") -> list[str]:
    if isinstance(value, Mapping):
        return [
            nested
            for key, item in value.items()
            for nested in _standalone_75_paths(item, f"{path}.{key}")
        ]
    if isinstance(value, list):
        return [
            nested
            for index, item in enumerate(value)
            for nested in _standalone_75_paths(item, f"{path}[{index}]")
        ]
    if isinstance(value, str) and re.search(r"(?<!\d)75(?!\d)", value):
        return [path]
    if not isinstance(value, bool) and isinstance(value, (int, float)) and value == 75:
        return [path]
    return []


def validate_guide_payload(
    payload: Mapping[str, Any], *, path: str = "html_v5/run1555_iteration_knowledge_guide.html"
) -> list[Failure]:
    """Validate the compact Guide domain without accepting root audit rows or inferred values."""
    failures: list[Failure] = []
    if not isinstance(payload, Mapping):
        return [_failure("GUIDE_CONTRACT", path, "Guide payload root must be an object")]
    violations: list[str] = []
    if set(payload) != _GUIDE_ROOT_KEYS:
        violations.append("root keys")
    if payload.get("schema_version") != "run1555-iteration-guide-v1":
        violations.append("schema")
    try:
        scope = payload["scope"]
        if (
            scope.get("direct_iterations") != DIRECT_PROFILER_ITERS
            or scope.get("direct_count") != 24
            or scope.get("inferred_excluded_count") != 75
        ):
            violations.append("scope")
        objects = payload["objects"]
        if not isinstance(objects, list) or [row.get("name") for row in objects] != _GUIDE_OBJECT_NAMES:
            violations.append("objects")
        steps = payload["steps"]
        if (
            not isinstance(steps, list)
            or len(steps) != 9
            or [row.get("order") for row in steps] != list(range(1, 10))
            or len({row.get("key") for row in steps}) != 9
        ):
            violations.append("steps")
        primitives = payload["primitives"]
        if not isinstance(primitives, list) or len(primitives) != 5 or len({row.get("name") for row in primitives}) != 5:
            violations.append("primitives")
        concepts = payload["concepts"]
        if not isinstance(concepts, list) or len(concepts) != 6 or len({row.get("name") for row in concepts}) != 6:
            violations.append("concepts")
        if set(payload["cases"]) != {"typical", "iter24", "iter88"}:
            violations.append("cases")
        quiz = payload["quiz"]
        if not isinstance(quiz, list) or len(quiz) != 7:
            violations.append("quiz")
        if {"data", "kpis", "profiler_view", "inference"} & set(payload):
            violations.append("root rows")
        allowed_75_paths = {
            "$.scope.inferred_excluded_count",
            "$.scope.evidence_boundary",
            "$.boundaries.direct_evidence",
        }
        if set(_standalone_75_paths(payload)) != allowed_75_paths:
            violations.append("75 boundary")
        if not _finite_tree(payload):
            violations.append("finite JSON")
    except (KeyError, TypeError, ValueError) as exc:
        violations.append(f"structure: {exc}")
    if violations:
        failures.append(
            _failure(
                "GUIDE_CONTRACT",
                path,
                f"Guide strict24 compact contract violations: {sorted(set(violations))}",
            )
        )
    return failures


@dataclass(frozen=True, order=True)
class CleanupCandidate:
    path: str
    kind: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class CheckOnlyResult:
    status: str
    mode: str
    passed_checks: int
    failed_checks: int
    failures: tuple[Failure, ...]
    manifest: str | None = None
    candidates: tuple[CleanupCandidate, ...] = ()
    deleted: tuple[CleanupCandidate, ...] = ()

    @property
    def exit_code(self) -> int:
        if self.status == "PASS":
            return 0
        operational_codes = {"INPUT_PATH_UNSAFE", "INPUT_MISSING", "WORKSPACE_PATH_UNSAFE", "MANIFEST_PATH_UNSAFE", "MANIFEST_WRITE"}
        return 2 if any(failure.code in operational_codes for failure in self.failures) else 1

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": "run1555-html-validator-result-v1",
            "status": self.status,
            "mode": self.mode,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "failures": [failure.to_dict() for failure in self.failures],
        }
        if self.manifest is not None:
            result["manifest"] = self.manifest
        if self.mode.startswith("cleanup-"):
            result["candidates"] = [candidate.to_dict() for candidate in self.candidates]
            result["deleted"] = [candidate.to_dict() for candidate in self.deleted]
        return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_row_count(path: Path) -> int:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = sum(1 for _ in csv.reader(handle))
        return max(0, rows - 1)
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    return 1


def _safe_workspace_path(workspace: Path, raw_path: str) -> tuple[Path | None, Failure | None]:
    root = workspace.resolve()
    relative = PurePosixPath(raw_path)
    if (
        not raw_path
        or "\\" in raw_path
        or relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() != raw_path
    ):
        return None, _failure(
            "INPUT_PATH_UNSAFE",
            raw_path or "<empty>",
            "input path must be a normalized relative POSIX path inside the workspace",
        )
    candidate = root.joinpath(*relative.parts)
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError:
        return None, _failure("INPUT_MISSING", raw_path, "metadata input file does not exist")
    if not resolved.is_relative_to(root):
        return None, _failure(
            "INPUT_PATH_UNSAFE",
            raw_path,
            "input path resolves outside the workspace (including an escaping symlink)",
        )
    if not resolved.is_file():
        return None, _failure("INPUT_MISSING", raw_path, "metadata input must resolve to a regular file")
    return resolved, None


def validate_input_metadata(
    payload: Mapping[str, Any], *, workspace: str | Path
) -> list[Failure]:
    """Validate each canonical input record against the current ordinary file."""
    failures: list[Failure] = []
    root = Path(workspace)
    metadata = payload.get("metadata") if isinstance(payload, Mapping) else None
    records = metadata.get("inputs") if isinstance(metadata, Mapping) else None
    if not isinstance(records, list):
        return [
            _failure(
                "INPUT_METADATA_STRUCTURE",
                "derived_v5/run1555_html_payload_v5.json",
                "metadata.inputs must be an array",
            )
        ]
    seen_paths: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            failures.append(
                _failure(
                    "INPUT_METADATA_STRUCTURE",
                    f"metadata.inputs[{index}]",
                    "input record must be an object",
                )
            )
            continue
        raw_path = record.get("path")
        if not isinstance(raw_path, str):
            failures.append(
                _failure(
                    "INPUT_METADATA_STRUCTURE",
                    f"metadata.inputs[{index}]",
                    "input record path must be a string",
                )
            )
            continue
        if raw_path in seen_paths:
            failures.append(_failure("INPUT_METADATA_STRUCTURE", raw_path, "input paths must be unique"))
            continue
        seen_paths.add(raw_path)
        source, path_failure = _safe_workspace_path(root, raw_path)
        if path_failure is not None:
            failures.append(path_failure)
            continue
        assert source is not None
        actual_size = source.stat().st_size
        if record.get("size_bytes") != actual_size:
            failures.append(
                _failure(
                    "INPUT_SIZE_BYTES",
                    raw_path,
                    f"metadata size_bytes is stale; expected {actual_size}",
                )
            )
        actual_rows = _source_row_count(source)
        if record.get("row_count") != actual_rows:
            failures.append(
                _failure(
                    "INPUT_ROW_COUNT",
                    raw_path,
                    f"metadata row_count is stale; expected {actual_rows}",
                )
            )
        actual_sha = _sha256(source)
        if record.get("sha256") != actual_sha:
            failures.append(
                _failure(
                    "INPUT_SHA256",
                    raw_path,
                    f"metadata sha256 is stale; expected {actual_sha}",
                )
            )
    return _sorted(failures)


_RUNTIME_NETWORK = re.compile(
    r"\b(?:fetch\s*\(|XMLHttpRequest\b|WebSocket\s*\(|EventSource\s*\()"
    r"|\bimport\s*\(\s*['\"](?:https?:)?//",
    re.IGNORECASE,
)
_SUCCESS_FALLBACK_HIDE = re.compile(
    r"\.js-plotly-plot\s*>\s*\.plot-fallback\s*\{[^}]*\bdisplay\s*:\s*none\s*;?",
    re.IGNORECASE | re.DOTALL,
)


def validate_plotly_contract(
    dashboard_html: str,
    *,
    css_text: str,
    plotly_text: str,
    ordered_scripts: Sequence[tuple[str, str]],
    path: str = "html_v5/run1555_v5_analysis_dashboard.html",
) -> list[Failure]:
    """Validate local Plotly provenance, embedding order, owned runtime, and fallback state."""
    failures: list[Failure] = []
    if "plotly.py 6.7.0" not in plotly_text or "plotly.js v3.5.0" not in plotly_text:
        failures.append(
            _failure(
                "PLOTLY_VERSION",
                "html_v5/assets/plotly-6.7.0.min.js",
                "bundle must identify plotly.py 6.7.0 and plotly.js v3.5.0 separately",
            )
        )
    if [name for name, _ in ordered_scripts] != [
        "shell",
        "vendor",
        "overview",
        "spatial",
        "headroom",
    ]:
        failures.append(
            _failure(
                "PLOTLY_INLINE_ORDER",
                path,
                "declared inline order must be shell→vendor→overview→spatial→headroom",
            )
        )
    soup = BeautifulSoup(dashboard_html, "html.parser")
    executable_scripts = [
        script
        for script in soup.find_all("script")
        if str(script.get("type", "")).lower() != "application/json"
    ]
    expected_script = "\n".join(text for _, text in ordered_scripts)
    if (
        len(executable_scripts) != 1
        or (executable_scripts[0].string or executable_scripts[0].get_text()).strip()
        != expected_script.strip()
    ):
        failures.append(
            _failure(
                "PLOTLY_INLINE_ORDER",
                path,
                "Dashboard executable script must exactly embed current shell/vendor/overview/spatial/headroom sources in order",
            )
        )
    styles = soup.find_all("style")
    if len(styles) != 1 or (styles[0].string or styles[0].get_text()).strip() != css_text.strip():
        failures.append(
            _failure(
                "DASHBOARD_CSS_EMBEDDING",
                path,
                "Dashboard must exactly embed the current owned CSS",
            )
        )
    plots = soup.select(".plot")
    if not plots or any(plot.select_one(".plot-fallback") is None for plot in plots):
        failures.append(
            _failure(
                "PLOT_FALLBACK",
                path,
                "every initial Plotly container must contain a .plot-fallback",
            )
        )
    if not _SUCCESS_FALLBACK_HIDE.search(css_text):
        failures.append(
            _failure(
                "PLOT_FALLBACK_HIDE_RULE",
                "html_v5/assets/design_tokens.css",
                "owned CSS must hide direct .plot-fallback children after Plotly success",
            )
        )
    for name, script_text in ordered_scripts:
        if name == "vendor":
            continue
        if _RUNTIME_NETWORK.search(script_text):
            failures.append(
                _failure(
                    "OWNED_JS_REMOTE_RUNTIME",
                    f"html_v5/assets/{name}.js",
                    "owned JavaScript must not use remote fetch/XHR/WebSocket/EventSource/import",
                )
            )
    return _sorted(failures)


def _embedded_payload(text: str, *, path: str) -> tuple[Mapping[str, Any] | None, list[Failure]]:
    soup = BeautifulSoup(text, "html.parser")
    tags = soup.select('script#run1555-payload[type="application/json"]')
    if len(tags) != 1:
        return None, [
            _failure(
                "HTML_PAYLOAD_COUNT",
                path,
                "exactly one application/json script#run1555-payload is required",
            )
        ]
    try:
        parsed = strict_json_loads(tags[0].string or tags[0].get_text())
    except ValueError as exc:
        return None, [_failure("HTML_PAYLOAD_JSON", path, str(exc))]
    if not isinstance(parsed, Mapping):
        return None, [_failure("HTML_PAYLOAD_JSON", path, "embedded payload root must be an object")]
    return parsed, []


def _fixed_workspace_file(root: Path, relative: str) -> Path:
    candidate = root / relative
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise RuntimeError(f"WORKSPACE_PATH_UNSAFE:{relative}")
    return resolved


def run_check_only(workspace: str | Path) -> CheckOnlyResult:
    """Read and validate current artifacts; this function performs no writes."""
    root = Path(workspace).resolve(strict=True)
    failures: list[Failure] = []
    passed_checks = 0

    def add(check_failures: Sequence[Failure]) -> None:
        nonlocal passed_checks
        if check_failures:
            failures.extend(check_failures)
        else:
            passed_checks += 1

    canonical_rel = "derived_v5/run1555_html_payload_v5.json"
    dashboard_rel = "html_v5/run1555_v5_analysis_dashboard.html"
    guide_rel = "html_v5/run1555_iteration_knowledge_guide.html"
    markdown_rel = "reports/HWN_GUIDE_20260714_Run1555_一个Iteration的计算与通信.md"
    canonical_path = _fixed_workspace_file(root, canonical_rel)
    dashboard_path = _fixed_workspace_file(root, dashboard_rel)
    guide_path = _fixed_workspace_file(root, guide_rel)
    markdown_path = _fixed_workspace_file(root, markdown_rel)

    try:
        canonical = strict_json_loads(canonical_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        canonical = None
        add([_failure("CANONICAL_JSON", canonical_rel, str(exc))])
    if not isinstance(canonical, Mapping):
        if canonical is not None:
            add([_failure("CANONICAL_JSON", canonical_rel, "canonical root must be an object")])
    else:
        add(validate_canonical_payload(canonical, path=canonical_rel))
        add(validate_input_metadata(canonical, workspace=root))

    dashboard_text = dashboard_path.read_text(encoding="utf-8")
    guide_text = guide_path.read_text(encoding="utf-8")
    add(validate_html_document(dashboard_text, path=dashboard_rel, product="dashboard"))
    add(validate_html_document(guide_text, path=guide_rel, product="guide"))
    add(validate_size_bytes(dashboard_path.stat().st_size, limit=16 * 1024 * 1024, path=dashboard_rel))
    add(validate_size_bytes(guide_path.stat().st_size, limit=1024 * 1024, path=guide_rel))

    dashboard_embedded, dashboard_parse_failures = _embedded_payload(
        dashboard_text, path=dashboard_rel
    )
    add(dashboard_parse_failures)
    guide_embedded, guide_parse_failures = _embedded_payload(guide_text, path=guide_rel)
    add(guide_parse_failures)

    if dashboard_embedded is not None:
        add(validate_dashboard_projection(dashboard_embedded, path=dashboard_rel))
    if guide_embedded is not None:
        add(validate_guide_payload(guide_embedded, path=guide_rel))

    if isinstance(canonical, Mapping) and dashboard_embedded is not None:
        from .html_render import dashboard_payload

        expected_dashboard = dashboard_payload(canonical)
        add(
            []
            if dashboard_embedded == expected_dashboard
            else [
                _failure(
                    "DASHBOARD_EMBEDDED_MISMATCH",
                    dashboard_rel,
                    "embedded payload must exactly equal dashboard_payload(current canonical)",
                )
            ]
        )

    from .iteration_knowledge import (
        build_iteration_guide_payload,
        build_iteration_knowledge_model,
        render_iteration_knowledge_markdown,
    )

    model = build_iteration_knowledge_model(root, payload=dict(canonical)) if isinstance(canonical, Mapping) else None
    if model is not None:
        expected_guide = build_iteration_guide_payload(model)
        if guide_embedded is not None:
            add(
                []
                if guide_embedded == expected_guide
                else [
                    _failure(
                        "GUIDE_EMBEDDED_MISMATCH",
                        guide_rel,
                        "embedded payload must exactly equal the current Task-5 model projection",
                    )
                ]
            )
        expected_markdown = render_iteration_knowledge_markdown(model)
        actual_markdown = markdown_path.read_text(encoding="utf-8")
        add(
            []
            if actual_markdown == expected_markdown
            else [
                _failure(
                    "GUIDE_MARKDOWN_MISMATCH",
                    markdown_rel,
                    "Markdown must exactly equal render_iteration_knowledge_markdown(current model)",
                )
            ]
        )

    assets = root / "html_v5/assets"
    css_text = _fixed_workspace_file(root, "html_v5/assets/design_tokens.css").read_text(
        encoding="utf-8"
    )
    ordered_asset_paths = [
        ("shell", "html_v5/assets/dashboard_shell.js"),
        ("vendor", "html_v5/assets/plotly-6.7.0.min.js"),
        ("overview", "html_v5/assets/dashboard_overview.js"),
        ("spatial", "html_v5/assets/dashboard_spatial.js"),
        ("headroom", "html_v5/assets/dashboard_headroom.js"),
    ]
    del assets
    ordered_scripts = [
        (name, _fixed_workspace_file(root, relative).read_text(encoding="utf-8"))
        for name, relative in ordered_asset_paths
    ]
    add(
        validate_plotly_contract(
            dashboard_text,
            css_text=css_text,
            plotly_text=ordered_scripts[1][1],
            ordered_scripts=ordered_scripts,
            path=dashboard_rel,
        )
    )
    guide_js = _fixed_workspace_file(root, "html_v5/assets/guide.js").read_text(encoding="utf-8")
    add(
        []
        if not _RUNTIME_NETWORK.search(guide_js)
        else [
            _failure(
                "OWNED_JS_REMOTE_RUNTIME",
                "html_v5/assets/guide.js",
                "owned Guide JavaScript must not use remote runtime APIs",
            )
        ]
    )

    final_html_names = sorted(path.name for path in (root / "html_v5").glob("*.html"))
    expected_names = sorted([Path(dashboard_rel).name, Path(guide_rel).name])
    add(
        []
        if final_html_names == expected_names
        else [
            _failure(
                "FINAL_HTML_DIRECTORY",
                "html_v5",
                f"top-level final HTML files must be exactly {expected_names}; got {final_html_names}",
            )
        ]
    )

    ordered_failures = tuple(_sorted(failures))
    return CheckOnlyResult(
        status="FAIL" if ordered_failures else "PASS",
        mode="check-only",
        passed_checks=passed_checks,
        failed_checks=len(ordered_failures),
        failures=ordered_failures,
    )


_MANIFEST_RELATIVE = "html_v5/manifest.json"
_MANIFEST_SCHEMA = "run1555-html-artifacts-manifest-v1"
_MANIFEST_SOURCE_PATHS = (
    "scripts/60_build_run1555_html_payload.py",
    "scripts/61_render_run1555_dashboard.py",
    "scripts/62_render_run1555_iteration_guide.py",
    "scripts/63_validate_html_artifacts.py",
    "src/x10000_analysis/html_validation.py",
    "src/x10000_analysis/html_payload.py",
    "src/x10000_analysis/html_render.py",
    "src/x10000_analysis/iteration_knowledge.py",
    "src/x10000_analysis/config.py",
    "src/x10000_analysis/headroom.py",
    "src/x10000_analysis/spatiotemporal.py",
    "html_v5/templates/run1555_analysis_dashboard.html",
    "html_v5/templates/run1555_iteration_guide.html",
    "html_v5/assets/design_tokens.css",
    "html_v5/assets/dashboard_shell.js",
    "html_v5/assets/dashboard_overview.js",
    "html_v5/assets/dashboard_spatial.js",
    "html_v5/assets/dashboard_headroom.js",
    "html_v5/assets/guide.js",
    "html_v5/assets/plotly-6.7.0.min.js",
    "package.json",
    "package-lock.json",
)
_SCREENSHOT_DIMENSIONS = {
    "html_v5/screenshots/dashboard_1920x1080.png": (1920, 1080),
    "html_v5/screenshots/dashboard_1440x900.png": (1440, 900),
    "html_v5/screenshots/guide_1440x900.png": (1440, 900),
    "html_v5/screenshots/guide_1024x768.png": (1024, 768),
}
_FULL_AUDIT_COUNTS = {
    "iter_summary": 99,
    "stage_share": 1188,
    "phase_interval_groups": 99,
    "phase_interval_rows": 35242,
    "stage_headroom": 891,
    "joint_counterfactual": 24,
    "collective_calls": 840,
    "nic_device_iter_rows": 792,
    "mtlink_gpu_iter_rows": 1584,
    "deepep_iter_summary_rows": 198,
}
_DASHBOARD_COUNTS = {
    "iter_summary": 24,
    "stage_share": 288,
    "phase_interval_groups": 24,
    "phase_interval_rows": 31074,
    "stage_headroom": 216,
    "joint_counterfactual": 24,
    "collective_calls": 840,
    "edp_entity_summary": 24,
    "edp_direct_summary": 16,
    "nic_device_iter_rows": 192,
    "nic_entity_summary_rows_each": 8,
    "mtlink_gpu_iter_rows": 384,
    "mtlink_entity_summary": 16,
    "deepep_iter_summary_rows": 48,
    "deepep_entity_summary": 32,
}
_PYTHON_TEST_ARGV = [
    "python3",
    "-m",
    "pytest",
    "-p",
    "no:cacheprovider",
    "tests_v5",
    "tests_html_v5",
    "-q",
]
_NODE_ARGV = [
    ["node", "--check", f"html_v5/assets/{name}"]
    for name in (
        "dashboard_shell.js",
        "dashboard_overview.js",
        "dashboard_spatial.js",
        "dashboard_headroom.js",
        "guide.js",
    )
] + [
    ["node", f"tests_html_v5/{name}"]
    for name in (
        "js_shell_behavior_test.js",
        "js_dashboard_overview_behavior_test.js",
        "js_dashboard_spatial_behavior_test.js",
        "js_dashboard_headroom_behavior_test.js",
        "js_guide_behavior_test.js",
    )
]
_BROWSER_ARGV = ["node", "tests_html_v5/playwright_task7_qa.mjs", "--capture-screenshots"]


class ManifestInvariantError(ValueError):
    """A deterministic manifest invariant is not satisfied by current files."""


def manifest_content_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash canonical compact JSON with the self-hash field omitted."""
    content = dict(manifest)
    content.pop("manifest_content_sha256", None)
    encoded = (
        json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            manifest,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def _manifest_file_entry(root: Path, relative: str) -> dict[str, Any]:
    try:
        path = _fixed_workspace_file(root, relative)
    except (FileNotFoundError, RuntimeError) as exc:
        raise ManifestInvariantError(f"required ordinary workspace file is missing or unsafe: {relative}") from exc
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ManifestInvariantError(f"not a PNG with an IHDR header: {path.name}")
    return struct.unpack(">II", header[16:24])


def _manifest_test_source_paths(root: Path) -> list[str]:
    paths: list[str] = []
    for directory in ("tests_v5", "tests_html_v5"):
        base = root / directory
        if not base.is_dir():
            raise ManifestInvariantError(f"test source directory is missing: {directory}")
        for suffix in ("*.py", "*.js", "*.mjs"):
            for candidate in base.rglob(suffix):
                if not candidate.is_file():
                    continue
                try:
                    resolved = candidate.resolve(strict=True)
                except FileNotFoundError as exc:
                    raise ManifestInvariantError(f"test source disappeared: {candidate}") from exc
                if not resolved.is_relative_to(root):
                    raise ManifestInvariantError(f"test source resolves outside workspace: {candidate}")
                paths.append(candidate.relative_to(root).as_posix())
    return sorted(set(paths))


def _manifest_inputs(root: Path, canonical: Mapping[str, Any]) -> list[dict[str, Any]]:
    metadata = canonical.get("metadata")
    records = metadata.get("inputs") if isinstance(metadata, Mapping) else None
    if not isinstance(records, list) or len(records) != 21:
        raise ManifestInvariantError("canonical metadata.inputs must contain exactly 21 records")
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            raise ManifestInvariantError("canonical metadata input records require a string path")
        relative = record["path"]
        if relative in seen:
            raise ManifestInvariantError(f"duplicate canonical metadata input path: {relative}")
        seen.add(relative)
        entry = _manifest_file_entry(root, relative)
        expected_bytes = record.get("size_bytes")
        expected_sha = record.get("sha256")
        row_count = record.get("row_count")
        if (
            isinstance(row_count, bool)
            or not isinstance(row_count, int)
            or row_count < 0
            or entry["bytes"] != expected_bytes
            or entry["sha256"] != expected_sha
        ):
            raise ManifestInvariantError(f"canonical metadata input facts are stale: {relative}")
        entry["row_count"] = row_count
        entries.append(entry)
    return sorted(entries, key=lambda item: item["path"])


def build_manifest(
    workspace: str | Path,
    *,
    browser_driver: str,
    accessibility_engine: str,
    slop_score: int,
    slop_3: int,
    slop_8: int,
    slop_10: int,
) -> dict[str, Any]:
    """Build the current deterministic PASS manifest entirely in memory."""
    root = Path(workspace).resolve(strict=True)
    if browser_driver != "playwright":
        raise ManifestInvariantError("browser_driver must equal playwright")
    if accessibility_engine not in {"axe", "playwright-dom-cdp"}:
        raise ManifestInvariantError("unsupported accessibility engine")
    if (
        isinstance(slop_score, bool)
        or not isinstance(slop_score, int)
        or not 0 <= slop_score <= 1
        or any(isinstance(value, bool) or not isinstance(value, int) or value != 0 for value in (slop_3, slop_8, slop_10))
    ):
        raise ManifestInvariantError("visual slop must satisfy score<=1 and items 3/8/10=0")

    canonical_path = _fixed_workspace_file(root, "derived_v5/run1555_html_payload_v5.json")
    canonical = strict_json_loads(canonical_path.read_text(encoding="utf-8"))
    if not isinstance(canonical, Mapping):
        raise ManifestInvariantError("canonical payload root must be an object")

    sources = sorted(
        (_manifest_file_entry(root, relative) for relative in _MANIFEST_SOURCE_PATHS),
        key=lambda item: item["path"],
    )
    tests = [
        _manifest_file_entry(root, relative)
        for relative in _manifest_test_source_paths(root)
    ]
    inputs = _manifest_inputs(root, canonical)

    canonical_entry = _manifest_file_entry(root, "derived_v5/run1555_html_payload_v5.json")
    canonical_entry["counts"] = dict(_FULL_AUDIT_COUNTS)
    dashboard_entry = _manifest_file_entry(root, "html_v5/run1555_v5_analysis_dashboard.html")
    dashboard_entry.update({"max_bytes_exclusive": 16 * 1024 * 1024, "counts": dict(_DASHBOARD_COUNTS)})
    markdown_entry = _manifest_file_entry(
        root, "reports/HWN_GUIDE_20260714_Run1555_一个Iteration的计算与通信.md"
    )
    guide_entry = _manifest_file_entry(root, "html_v5/run1555_iteration_knowledge_guide.html")
    guide_entry.update(
        {
            "max_bytes_exclusive": 1024 * 1024,
            "counts": {
                "objects": 5,
                "steps": 9,
                "primitives": 5,
                "concepts": 6,
                "cases": 3,
                "quiz": 7,
            },
        }
    )

    screenshot_dir = root / "html_v5/screenshots"
    actual_pngs = (
        sorted(path.relative_to(root).as_posix() for path in screenshot_dir.glob("*.png"))
        if screenshot_dir.is_dir()
        else []
    )
    if actual_pngs != sorted(_SCREENSHOT_DIMENSIONS):
        raise ManifestInvariantError(
            f"screenshot paths must be exact; expected {sorted(_SCREENSHOT_DIMENSIONS)}, got {actual_pngs}"
        )
    screenshots: list[dict[str, Any]] = []
    for relative, expected_dimensions in _SCREENSHOT_DIMENSIONS.items():
        entry = _manifest_file_entry(root, relative)
        dimensions = _png_dimensions(root / relative)
        if dimensions != expected_dimensions:
            raise ManifestInvariantError(
                f"PNG dimensions are wrong for {relative}: expected {expected_dimensions}, got {dimensions}"
            )
        entry.update({"width": dimensions[0], "height": dimensions[1]})
        screenshots.append(entry)
    screenshots.sort(key=lambda item: item["path"])

    manifest: dict[str, Any] = {
        "schema_version": _MANIFEST_SCHEMA,
        "hash_algorithm": "sha256",
        "status": "PASS",
        "schemas": {
            "canonical_payload": "run1555-html-payload-v5.2",
            "dashboard_embedded_payload": "run1555-html-payload-v5.2",
            "guide_embedded_payload": "run1555-iteration-guide-v1",
        },
        "scope": {
            "mode": "direct_profiler_only",
            "membership_source": "run.coverage.direct_profiler_iters",
            "iterations": list(DIRECT_PROFILER_ITERS),
            "iter_count": 24,
            "excluded_inference_count": 75,
            "iter100_excluded": True,
        },
        "audit": {
            "canonical_iter_count": 99,
            "validated_inference_count": 75,
            "full_audit_counts": dict(_FULL_AUDIT_COUNTS),
        },
        "plotly": {
            "python_package_version": "6.7.0",
            "javascript_bundle_version": "3.5.0",
            "asset": _manifest_file_entry(root, "html_v5/assets/plotly-6.7.0.min.js"),
        },
        "artifacts": {
            "canonical_payload": canonical_entry,
            "dashboard": dashboard_entry,
            "guide_markdown": markdown_entry,
            "guide_html": guide_entry,
        },
        "sources": sources,
        "test_sources": tests,
        "inputs": inputs,
        "validation": {
            "static": {
                "status": "PASS",
                "command": ["python3", "scripts/63_validate_html_artifacts.py", "--check-only"],
                "exit_code": 0,
            },
            "python_tests": {"status": "PASS", "command": list(_PYTHON_TEST_ARGV), "exit_code": 0},
            "node": {"status": "PASS", "commands": [list(argv) for argv in _NODE_ARGV], "exit_code": 0},
            "browser": {
                "status": "PASS",
                "driver": browser_driver,
                "command": list(_BROWSER_ARGV),
                "exit_code": 0,
                "console_errors": 0,
                "console_warnings": 0,
                "external_requests": 0,
            },
            "visual": {
                "status": "PASS",
                "attestation_argv": [
                    "--slop-score",
                    str(slop_score),
                    "--slop-3",
                    str(slop_3),
                    "--slop-8",
                    str(slop_8),
                    "--slop-10",
                    str(slop_10),
                ],
                "slop_max_score": slop_score,
                "slop_item_3": slop_3,
                "slop_item_8": slop_8,
                "slop_item_10": slop_10,
            },
            "accessibility": {
                "status": "PASS",
                "engine": accessibility_engine,
                "command": list(_BROWSER_ARGV),
                "blocking_violations": 0,
            },
        },
        "screenshots": screenshots,
        "cleanup": {
            "status": "PASS",
            "forbidden_matches": 0,
            "duplicate_final_html_groups": 0,
        },
    }
    manifest["manifest_content_sha256"] = manifest_content_sha256(manifest)
    return manifest


def atomic_write_manifest(workspace: str | Path, manifest: Mapping[str, Any]) -> Path:
    """Atomically replace html_v5/manifest.json with deterministic pretty JSON."""
    root = Path(workspace).resolve(strict=True)
    directory = (root / "html_v5").resolve(strict=True)
    if not directory.is_relative_to(root) or not directory.is_dir():
        raise RuntimeError("MANIFEST_PATH_UNSAFE:html_v5")
    destination = directory / "manifest.json"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=directory,
            prefix=".manifest.json.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(_manifest_bytes(manifest))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
        try:
            directory_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _remove_old_manifest(root: Path) -> None:
    manifest_path = root / _MANIFEST_RELATIVE
    if manifest_path.is_symlink() or manifest_path.is_file():
        manifest_path.unlink()
    elif manifest_path.exists():
        raise RuntimeError(f"MANIFEST_PATH_UNSAFE:{_MANIFEST_RELATIVE}")


def _finalize_failure(
    root: Path,
    *,
    passed_checks: int,
    failures: Sequence[Failure],
) -> CheckOnlyResult:
    _remove_old_manifest(root)
    ordered = tuple(_sorted(list(failures)))
    return CheckOnlyResult(
        status="FAIL",
        mode="finalize",
        passed_checks=passed_checks,
        failed_checks=len(ordered),
        failures=ordered,
        manifest=_MANIFEST_RELATIVE,
    )


def _effective_task_tmp_root(root: Path, explicit: str | Path | None) -> Path:
    """Select global Task 7 temp cleanup only for the real project workspace.

    Unit-test and copied workspaces must not inherit unrelated `/tmp/run1555_task7_*`
    files from concurrent sessions. Callers can still inject an explicit temp root.
    """
    if explicit is not None:
        return Path(explicit)
    project_root = Path(__file__).resolve().parents[2]
    return Path("/tmp") if root == project_root else root / ".run1555-task-tmp-disabled"


def finalize_manifest(
    workspace: str | Path,
    *,
    attest_python_tests_pass: bool = False,
    attest_node_pass: bool = False,
    attest_browser_pass: bool = False,
    attest_visual_pass: bool = False,
    attest_accessibility_pass: bool = False,
    browser_driver: Any = None,
    accessibility_engine: Any = None,
    slop_score: Any = None,
    slop_3: Any = None,
    slop_8: Any = None,
    slop_10: Any = None,
    check_runner: Callable[[str | Path], CheckOnlyResult] = run_check_only,
    task_tmp_root: str | Path | None = None,
) -> CheckOnlyResult:
    """Run the read-only core, require all attestations, then atomically finalize."""
    root = Path(workspace).resolve(strict=True)
    core = check_runner(root)
    if core.status != "PASS":
        return _finalize_failure(root, passed_checks=core.passed_checks, failures=core.failures)

    attestations = {
        "--attest-python-tests-pass": attest_python_tests_pass,
        "--attest-node-pass": attest_node_pass,
        "--attest-browser-pass": attest_browser_pass,
        "--attest-visual-pass": attest_visual_pass,
        "--attest-accessibility-pass": attest_accessibility_pass,
    }
    missing = sorted(name for name, present in attestations.items() if present is not True)
    if missing:
        return _finalize_failure(
            root,
            passed_checks=core.passed_checks,
            failures=[
                _failure(
                    "FINALIZE_ATTESTATION",
                    _MANIFEST_RELATIVE,
                    f"missing required PASS attestations: {missing}",
                )
            ],
        )
    cleanup = run_cleanup(
        root,
        "verify",
        task_tmp_root=_effective_task_tmp_root(root, task_tmp_root),
    )
    if cleanup.status != "PASS":
        return _finalize_failure(
            root,
            passed_checks=core.passed_checks + len(attestations),
            failures=cleanup.failures,
        )
    try:
        manifest = build_manifest(
            root,
            browser_driver=browser_driver,
            accessibility_engine=accessibility_engine,
            slop_score=slop_score,
            slop_3=slop_3,
            slop_8=slop_8,
            slop_10=slop_10,
        )
    except (ManifestInvariantError, ValueError, FileNotFoundError) as exc:
        return _finalize_failure(
            root,
            passed_checks=core.passed_checks + len(attestations),
            failures=[_failure("MANIFEST_INVARIANT", _MANIFEST_RELATIVE, str(exc))],
        )
    try:
        atomic_write_manifest(root, manifest)
    except OSError as exc:
        _remove_old_manifest(root)
        return CheckOnlyResult(
            status="FAIL",
            mode="finalize",
            passed_checks=core.passed_checks + len(attestations) + 1,
            failed_checks=1,
            failures=(
                _failure("MANIFEST_WRITE", _MANIFEST_RELATIVE, f"atomic manifest write failed: {exc}"),
            ),
            manifest=_MANIFEST_RELATIVE,
        )
    return CheckOnlyResult(
        status="PASS",
        mode="finalize",
        passed_checks=core.passed_checks + len(attestations) + 1,
        failed_checks=0,
        failures=(),
        manifest=_MANIFEST_RELATIVE,
    )


_MANIFEST_TOP_KEYS = {
    "schema_version",
    "hash_algorithm",
    "manifest_content_sha256",
    "status",
    "schemas",
    "scope",
    "audit",
    "plotly",
    "artifacts",
    "sources",
    "test_sources",
    "inputs",
    "validation",
    "screenshots",
    "cleanup",
}
_FORBIDDEN_MANIFEST_KEYS = {
    "generated_at",
    "generated_at_utc",
    "validated_at",
    "capture_time",
    "captured_at",
    "timestamp",
    "mtime",
    "ctime",
    "inode",
    "duration",
    "duration_ms",
    "pid",
    "session",
    "session_id",
    "browser_session_id",
}


def _manifest_shape_failures(value: Mapping[str, Any]) -> list[Failure]:
    failures: list[Failure] = []
    if set(value) != _MANIFEST_TOP_KEYS:
        failures.append(
            _failure(
                "MANIFEST_SCHEMA",
                _MANIFEST_RELATIVE,
                f"manifest top-level keys must be exact; got {sorted(value)}",
            )
        )
    if value.get("schema_version") != _MANIFEST_SCHEMA or value.get("hash_algorithm") != "sha256":
        failures.append(
            _failure("MANIFEST_SCHEMA", _MANIFEST_RELATIVE, "manifest schema/hash algorithm is invalid")
        )
    if value.get("status") != "PASS":
        failures.append(_failure("MANIFEST_STATUS", _MANIFEST_RELATIVE, "manifest status must equal PASS"))
    supplied_hash = value.get("manifest_content_sha256")
    if not isinstance(supplied_hash, str) or supplied_hash != manifest_content_sha256(value):
        failures.append(
            _failure("MANIFEST_SELF_HASH", _MANIFEST_RELATIVE, "manifest content self-hash does not match")
        )

    def walk(nested: Any, location: str) -> None:
        if isinstance(nested, Mapping):
            for key, item in nested.items():
                lowered = str(key).lower()
                if lowered in _FORBIDDEN_MANIFEST_KEYS or lowered.endswith("_pid"):
                    failures.append(
                        _failure(
                            "MANIFEST_DRIFT_FIELD",
                            _MANIFEST_RELATIVE,
                            f"forbidden drifting key at {location}.{key}",
                        )
                    )
                if key == "path":
                    if (
                        not isinstance(item, str)
                        or not item
                        or "\\" in item
                        or PurePosixPath(item).is_absolute()
                        or ".." in PurePosixPath(item).parts
                        or PurePosixPath(item).as_posix() != item
                    ):
                        failures.append(
                            _failure(
                                "MANIFEST_PATH",
                                _MANIFEST_RELATIVE,
                                f"manifest path must be normalized relative POSIX at {location}",
                            )
                        )
                walk(item, f"{location}.{key}")
        elif isinstance(nested, list):
            for index, item in enumerate(nested):
                walk(item, f"{location}[{index}]")

    walk(value, "$")
    for key in ("sources", "test_sources", "inputs", "screenshots"):
        entries = value.get(key)
        if not isinstance(entries, list) or any(not isinstance(entry, Mapping) for entry in entries):
            failures.append(_failure("MANIFEST_SCHEMA", _MANIFEST_RELATIVE, f"{key} must be an array of objects"))
            continue
        paths = [entry.get("path") for entry in entries]
        if any(not isinstance(path, str) for path in paths) or paths != sorted(paths) or len(paths) != len(set(paths)):
            failures.append(
                _failure("MANIFEST_PATH_ARRAY", _MANIFEST_RELATIVE, f"{key} paths must be sorted and unique")
            )
    return _sorted(failures)


def verify_manifest(
    workspace: str | Path,
    *,
    check_runner: Callable[[str | Path], CheckOnlyResult] = run_check_only,
    task_tmp_root: str | Path | None = None,
) -> CheckOnlyResult:
    """Strictly parse and compare the manifest with the complete current deterministic view."""
    root = Path(workspace).resolve(strict=True)
    core = check_runner(root)
    failures: list[Failure] = list(core.failures) if core.status != "PASS" else []
    cleanup = run_cleanup(
        root,
        "verify",
        task_tmp_root=_effective_task_tmp_root(root, task_tmp_root),
    )
    if cleanup.status != "PASS":
        failures.extend(cleanup.failures)
    path = root / _MANIFEST_RELATIVE
    if not path.is_file() or path.is_symlink():
        failures.append(_failure("MANIFEST_MISSING", _MANIFEST_RELATIVE, "ordinary manifest file is required"))
    else:
        try:
            parsed = strict_json_loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            parsed = None
            failures.append(_failure("MANIFEST_JSON", _MANIFEST_RELATIVE, str(exc)))
        if parsed is not None and not isinstance(parsed, Mapping):
            failures.append(_failure("MANIFEST_JSON", _MANIFEST_RELATIVE, "manifest root must be an object"))
        elif isinstance(parsed, Mapping):
            shape_failures = _manifest_shape_failures(parsed)
            failures.extend(shape_failures)
            if not shape_failures:
                try:
                    validation = parsed["validation"]
                    browser = validation["browser"]
                    accessibility = validation["accessibility"]
                    visual = validation["visual"]
                    expected = build_manifest(
                        root,
                        browser_driver=browser["driver"],
                        accessibility_engine=accessibility["engine"],
                        slop_score=visual["slop_max_score"],
                        slop_3=visual["slop_item_3"],
                        slop_8=visual["slop_item_8"],
                        slop_10=visual["slop_item_10"],
                    )
                    if parsed != expected:
                        failures.append(
                            _failure(
                                "MANIFEST_STALE",
                                _MANIFEST_RELATIVE,
                                "manifest differs from current artifacts, sources, tests, inputs, screenshots, or fixed validation contract",
                            )
                        )
                except (KeyError, TypeError, ValueError, FileNotFoundError, ManifestInvariantError) as exc:
                    failures.append(_failure("MANIFEST_STALE", _MANIFEST_RELATIVE, str(exc)))
    ordered = tuple(_sorted(failures))
    return CheckOnlyResult(
        status="FAIL" if ordered else "PASS",
        mode="verify-manifest",
        passed_checks=core.passed_checks + (0 if ordered else 1),
        failed_checks=len(ordered),
        failures=ordered,
        manifest=_MANIFEST_RELATIVE,
    )


_CLEANUP_PROTECTED_PREFIXES = {
    ("html_v5", "assets"),
    ("html_v5", "templates"),
}


def _classify_cleanup_path(root: Path, path: Path, task_tmp_root: Path) -> CleanupCandidate | None:
    if path.is_symlink():
        return None
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError:
        return None

    if path.is_absolute() and not resolved.is_relative_to(root):
        try:
            tmp_resolved = task_tmp_root.resolve(strict=True)
        except FileNotFoundError:
            return None
        if (
            not resolved.is_relative_to(tmp_resolved)
            or resolved.parent != tmp_resolved
            or not resolved.name.startswith("run1555_task7_")
        ):
            return None
        kind = "task-temp-directory" if resolved.is_dir() else "task-temp-file" if resolved.is_file() else None
        return CleanupCandidate(str(resolved), kind) if kind is not None else None

    if not resolved.is_relative_to(root):
        return None
    try:
        lexical = path.relative_to(root)
        relative = lexical.as_posix()
        resolved_relative = resolved.relative_to(root)
    except ValueError:
        return None
    cursor = root
    for part in lexical.parts[:-1]:
        cursor = cursor / part
        if cursor.is_symlink():
            return None
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != relative:
        return None
    if (
        tuple(pure.parts[:2]) in _CLEANUP_PROTECTED_PREFIXES
        or tuple(resolved_relative.parts[:2]) in _CLEANUP_PROTECTED_PREFIXES
    ):
        return None

    name = path.name
    if relative == "node_modules" and resolved.is_dir():
        return CleanupCandidate(relative, "qa-node-modules-directory")
    if resolved.is_dir() and name in {"__pycache__", ".pytest_cache"}:
        return CleanupCandidate(relative, "cache-directory")
    if name.startswith("._"):
        kind = "appledouble-directory" if resolved.is_dir() else "appledouble-file" if resolved.is_file() else None
        return CleanupCandidate(relative, kind) if kind is not None else None
    if resolved.is_file() and name == ".DS_Store":
        return CleanupCandidate(relative, "ds-store-file")
    if resolved.is_file() and name.endswith(".pyc"):
        return CleanupCandidate(relative, "bytecode-file")
    if resolved.is_file() and pure.parts and pure.parts[0] == "html_v5" and name.endswith(".tmp"):
        return CleanupCandidate(relative, "html-temp-file")
    return None


def discover_cleanup_candidates(
    workspace: str | Path,
    *,
    task_tmp_root: str | Path = "/tmp",
) -> list[CleanupCandidate]:
    """Return a sorted, minimal allowlist plan without mutating anything."""
    root = Path(workspace).resolve(strict=True)
    tmp_root = Path(task_tmp_root)
    candidates: list[CleanupCandidate] = []
    for current_text, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        current = Path(current_text)
        directory_names.sort()
        file_names.sort()
        retained: list[str] = []
        for name in directory_names:
            path = current / name
            relative_parts = path.relative_to(root).parts
            if path.is_symlink() or tuple(relative_parts[:2]) in _CLEANUP_PROTECTED_PREFIXES:
                continue
            if name == "node_modules" and path != root / "node_modules":
                continue
            candidate = _classify_cleanup_path(root, path, tmp_root)
            if candidate is not None:
                candidates.append(candidate)
            else:
                retained.append(name)
        directory_names[:] = retained
        for name in file_names:
            candidate = _classify_cleanup_path(root, current / name, tmp_root)
            if candidate is not None:
                candidates.append(candidate)

    if tmp_root.is_dir() and not tmp_root.is_symlink():
        for path in sorted(tmp_root.iterdir(), key=lambda item: item.name):
            if not path.name.startswith("run1555_task7_"):
                continue
            candidate = _classify_cleanup_path(root, path, tmp_root)
            if candidate is not None:
                candidates.append(candidate)
    return sorted(set(candidates), key=lambda item: (item.path, item.kind))


def _cleanup_candidate_path(root: Path, candidate: CleanupCandidate) -> Path:
    raw = Path(candidate.path)
    return raw if raw.is_absolute() else root.joinpath(*PurePosixPath(candidate.path).parts)


def apply_cleanup_candidates(
    workspace: str | Path,
    candidates: Sequence[CleanupCandidate],
    *,
    task_tmp_root: str | Path = "/tmp",
) -> tuple[tuple[CleanupCandidate, ...], tuple[Failure, ...]]:
    """Re-resolve and re-match every planned candidate immediately before deletion."""
    root = Path(workspace).resolve(strict=True)
    tmp_root = Path(task_tmp_root)
    deleted: list[CleanupCandidate] = []
    failures: list[Failure] = []
    for planned in sorted(set(candidates), key=lambda item: (item.path, item.kind)):
        path = _cleanup_candidate_path(root, planned)
        current = _classify_cleanup_path(root, path, tmp_root)
        if current != planned:
            failures.append(
                _failure(
                    "CLEANUP_RECHECK",
                    planned.path,
                    "candidate no longer resolves to the same allowlisted path and kind",
                )
            )
            continue
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            deleted.append(planned)
        except OSError as exc:
            failures.append(_failure("CLEANUP_DELETE", planned.path, f"delete failed: {exc}"))
    return tuple(deleted), tuple(_sorted(failures))


def run_cleanup(
    workspace: str | Path,
    mode: str,
    *,
    task_tmp_root: str | Path = "/tmp",
) -> CheckOnlyResult:
    """Execute cleanup dry-run/apply/verify with fail-closed result semantics."""
    if mode not in {"dry-run", "apply", "verify"}:
        raise ValueError("cleanup mode must be dry-run, apply, or verify")
    root = Path(workspace).resolve(strict=True)
    candidates = tuple(discover_cleanup_candidates(root, task_tmp_root=task_tmp_root))
    if mode == "dry-run":
        return CheckOnlyResult("PASS", "cleanup-dry-run", 1, 0, (), candidates=candidates)
    if mode == "verify":
        failures = (
            (
                _failure(
                    "CLEANUP_NOT_CLEAN",
                    ".",
                    f"cleanup allowlist still has {len(candidates)} candidate(s)",
                ),
            )
            if candidates
            else ()
        )
        return CheckOnlyResult(
            "FAIL" if failures else "PASS",
            "cleanup-verify",
            0 if failures else 1,
            len(failures),
            failures,
            candidates=candidates,
        )

    deleted, failures = apply_cleanup_candidates(root, candidates, task_tmp_root=task_tmp_root)
    remaining = tuple(discover_cleanup_candidates(root, task_tmp_root=task_tmp_root))
    apply_failures = list(failures)
    if remaining:
        apply_failures.append(
            _failure(
                "CLEANUP_REMAINING",
                ".",
                f"cleanup apply left {len(remaining)} allowlisted candidate(s)",
            )
        )
    ordered = tuple(_sorted(apply_failures))
    return CheckOnlyResult(
        "FAIL" if ordered else "PASS",
        "cleanup-apply",
        0 if ordered else 1,
        len(ordered),
        ordered,
        candidates=candidates,
        deleted=deleted,
    )


class _CLIUsageError(ValueError):
    pass


class _JSONArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _CLIUsageError(message)


def _write_cli_result(result: CheckOnlyResult, stdout: TextIO) -> None:
    stdout.write(
        json.dumps(
            result.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    )
    stdout.flush()


def _cli_int(value: str | None) -> Any:
    if value is None:
        return None
    return int(value) if re.fullmatch(r"-?\d+", value) else value


def cli_main(
    argv: Sequence[str] | None = None,
    *,
    workspace: str | Path | None = None,
    runner: Callable[[str | Path], CheckOnlyResult] | None = None,
    stdout: TextIO | None = None,
) -> int:
    """Structured multi-mode CLI core used by the thin script and tests."""
    output = sys.stdout if stdout is None else stdout
    execute = run_check_only if runner is None else runner
    raw_argv = list(argv) if argv is not None else list(sys.argv[1:])
    mode_hint = (
        "finalize"
        if "--finalize" in raw_argv
        else "verify-manifest"
        if "--verify-manifest" in raw_argv
        else "cleanup"
        if "--cleanup" in raw_argv
        else "check-only"
    )
    try:
        parser = _JSONArgumentParser(prog="63_validate_html_artifacts.py", add_help=False)
        modes = parser.add_mutually_exclusive_group(required=True)
        modes.add_argument("--check-only", action="store_true")
        modes.add_argument("--finalize", action="store_true")
        modes.add_argument("--verify-manifest", action="store_true")
        modes.add_argument("--cleanup")
        parser.add_argument("--attest-python-tests-pass", action="store_true")
        parser.add_argument("--attest-node-pass", action="store_true")
        parser.add_argument("--attest-browser-pass", action="store_true")
        parser.add_argument("--attest-visual-pass", action="store_true")
        parser.add_argument("--attest-accessibility-pass", action="store_true")
        parser.add_argument("--browser-driver")
        parser.add_argument("--accessibility-engine")
        parser.add_argument("--slop-score")
        parser.add_argument("--slop-3")
        parser.add_argument("--slop-8")
        parser.add_argument("--slop-10")
        arguments = parser.parse_args(raw_argv)
        root = Path(__file__).resolve().parents[2] if workspace is None else Path(workspace)
        finalize_options_present = any(
            token == option or token.startswith(option + "=")
            for token in raw_argv
            for option in (
                "--attest-python-tests-pass",
                "--attest-node-pass",
                "--attest-browser-pass",
                "--attest-visual-pass",
                "--attest-accessibility-pass",
                "--browser-driver",
                "--accessibility-engine",
                "--slop-score",
                "--slop-3",
                "--slop-8",
                "--slop-10",
            )
        )
        if arguments.check_only:
            if finalize_options_present:
                raise _CLIUsageError("finalize options are only valid with --finalize")
            result = execute(root)
        elif arguments.finalize:
            result = finalize_manifest(
                root,
                attest_python_tests_pass=arguments.attest_python_tests_pass,
                attest_node_pass=arguments.attest_node_pass,
                attest_browser_pass=arguments.attest_browser_pass,
                attest_visual_pass=arguments.attest_visual_pass,
                attest_accessibility_pass=arguments.attest_accessibility_pass,
                browser_driver=arguments.browser_driver,
                accessibility_engine=arguments.accessibility_engine,
                slop_score=_cli_int(arguments.slop_score),
                slop_3=_cli_int(arguments.slop_3),
                slop_8=_cli_int(arguments.slop_8),
                slop_10=_cli_int(arguments.slop_10),
                check_runner=execute,
            )
        elif arguments.verify_manifest:
            if finalize_options_present:
                raise _CLIUsageError("finalize options are only valid with --finalize")
            result = verify_manifest(root, check_runner=execute)
        else:
            if arguments.cleanup not in {"dry-run", "apply", "verify"}:
                raise _CLIUsageError("--cleanup requires dry-run, apply, or verify")
            if finalize_options_present:
                raise _CLIUsageError("finalize options are only valid with --finalize")
            result = run_cleanup(root, arguments.cleanup)
        _write_cli_result(result, output)
        return result.exit_code
    except _CLIUsageError as exc:
        result = CheckOnlyResult(
            status="FAIL",
            mode=mode_hint,
            passed_checks=0,
            failed_checks=1,
            failures=(
                _failure(
                    "CLI_USAGE",
                    "scripts/63_validate_html_artifacts.py",
                    str(exc),
                ),
            ),
            manifest=_MANIFEST_RELATIVE if mode_hint in {"finalize", "verify-manifest"} else None,
        )
        _write_cli_result(result, output)
        return 2
    except Exception as exc:
        result = CheckOnlyResult(
            status="FAIL",
            mode=mode_hint,
            passed_checks=0,
            failed_checks=1,
            failures=(
                _failure(
                    "CLI_INTERNAL",
                    "scripts/63_validate_html_artifacts.py",
                    f"{type(exc).__name__}: {exc}",
                ),
            ),
            manifest=_MANIFEST_RELATIVE if mode_hint in {"finalize", "verify-manifest"} else None,
        )
        _write_cli_result(result, output)
        return 2
