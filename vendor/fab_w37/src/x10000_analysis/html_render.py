from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

_PLACEHOLDER = re.compile(r"{{\s*[A-Z][A-Z0-9_]*\s*}}")
_SHELL_KEYS = (
    "schema_version",
    "metadata",
    "validator",
    "run",
    "kpis",
    "stage_taxonomy",
)
_DASHBOARD_KEYS = (
    "schema_version",
    "metadata",
    "validator",
    "run",
    "stage_taxonomy",
    "semantics",
    "profiler_view",
)


def shell_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the compact Task 2 contract while allowing callers to pass full data later."""
    missing = [key for key in _SHELL_KEYS if key not in payload]
    if missing:
        raise ValueError(f"missing required shell payload keys: {', '.join(missing)}")
    return {key: payload[key] for key in _SHELL_KEYS}


def dashboard_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the profiler-only browser projection plus evidence-only audit metadata."""
    missing = [key for key in _DASHBOARD_KEYS if key not in payload]
    if missing:
        raise ValueError(f"missing required dashboard payload keys: {', '.join(missing)}")

    data = payload.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("dashboard projection requires root data for audit counts")
    spatial = data.get("spatial_temporal")
    if not isinstance(spatial, Mapping):
        raise ValueError("dashboard projection requires root data.spatial_temporal")

    def rows(container: Mapping[str, Any], key: str, label: str) -> list[Any]:
        value = container.get(key)
        if not isinstance(value, list):
            raise ValueError(f"dashboard projection requires {label} array")
        return value

    nic = spatial.get("nic")
    mtlink = spatial.get("mtlink")
    deepep = spatial.get("deepep")
    if not all(isinstance(section, Mapping) for section in (nic, mtlink, deepep)):
        raise ValueError("dashboard projection requires NIC, MTLink, and DeepEP audit sections")

    existing_evidence = payload.get("evidence", {})
    if not isinstance(existing_evidence, Mapping):
        raise ValueError("dashboard evidence must be an object when present")
    if "inference" not in data:
        raise ValueError("dashboard projection requires root data.inference evidence")

    evidence = dict(existing_evidence)
    evidence["inference"] = data["inference"]
    evidence["full_audit_counts"] = {
        "iter_count": len(rows(data, "iter_summary", "root data.iter_summary")),
        "stage_share_rows": len(rows(data, "stage_share", "root data.stage_share")),
        "stage_headroom_rows": len(rows(data, "stage_headroom", "root data.stage_headroom")),
        "nic_device_iter_rows": len(rows(nic, "per_iter_device", "root NIC per_iter_device")),
        "mtlink_gpu_iter_rows": len(rows(mtlink, "per_iter_gpu", "root MTLink per_iter_gpu")),
        "deepep_iter_summary_rows": len(rows(deepep, "iter_summary", "root DeepEP iter_summary")),
    }
    projected = {key: payload[key] for key in _DASHBOARD_KEYS}
    projected["evidence"] = evidence
    return projected


def render_html(
    template_text: str,
    *,
    css_text: str,
    js_text: str,
    payload: Mapping[str, Any],
) -> str:
    """Replace the shell's explicit placeholders with trusted local assets and JSON."""
    payload_json = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    # HTML's script-data parser is case-insensitive; preserve JSON value while
    # preventing payload text from terminating the application/json element.
    payload_json = re.sub(r"</script", r"<\\/script", payload_json, flags=re.IGNORECASE)
    replacements = {
        "{{INLINE_CSS}}": css_text,
        "{{INLINE_PAYLOAD}}": payload_json,
        "{{INLINE_JS}}": js_text,
    }
    rendered = template_text
    for placeholder, value in replacements.items():
        if placeholder not in rendered:
            raise ValueError(f"template is missing required placeholder {placeholder}")
        rendered = rendered.replace(placeholder, value)
    unresolved = _PLACEHOLDER.findall(rendered)
    if unresolved:
        raise ValueError(f"unresolved template placeholders: {unresolved}")
    return rendered


def render_paths(
    *,
    template_path: Path,
    css_path: Path,
    js_path: Path,
    output_path: Path,
    payload_path: Path | None = None,
    payload_override: Mapping[str, Any] | None = None,
    compact_payload: bool | None = None,
    payload_projection: Literal["full", "dashboard"] | None = None,
    extra_js_paths: Sequence[Path] = (),
) -> int:
    """Read local assets and atomically write a UTF-8, directly openable HTML file.

    Exactly one payload source is required. A direct override is embedded as-is and cannot be
    combined with the legacy compact/dashboard/full projection paths.
    """
    if (payload_path is None) == (payload_override is None):
        raise ValueError("exactly one of payload_path or payload_override must be provided")
    if payload_projection not in (None, "full", "dashboard"):
        raise ValueError(f"unsupported payload_projection: {payload_projection}")

    if payload_override is not None:
        if not isinstance(payload_override, Mapping):
            raise TypeError("payload_override must be a mapping")
        if compact_payload is True or payload_projection is not None:
            raise ValueError(
                "payload_override cannot be combined with compact or dashboard/full projection"
            )
        embedded: Mapping[str, Any] = payload_override
    else:
        effective_compact = True if compact_payload is None else compact_payload
        if effective_compact and payload_projection == "dashboard":
            raise ValueError("compact_payload and dashboard payload_projection cannot both be enabled")
        assert payload_path is not None
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        if payload_projection == "dashboard":
            embedded = dashboard_payload(payload)
        elif payload_projection == "full":
            embedded = payload
        else:
            embedded = shell_payload(payload) if effective_compact else payload
    js_text = "\n".join(
        path.read_text(encoding="utf-8") for path in (js_path, *extra_js_paths)
    )
    rendered = render_html(
        template_path.read_text(encoding="utf-8"),
        css_text=css_path.read_text(encoding="utf-8"),
        js_text=js_text,
        payload=embedded,
    )
    encoded = rendered.encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(encoded)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_name = temporary.name
        os.replace(temporary_name, output_path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return len(encoded)
