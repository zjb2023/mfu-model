"""One-time hash-checked import; leaves the source tree and seals untouched.

The audited set contains derived data, code and reports, not raw traces.
Copies are deliberately ordinary files, never hard links to mutable sources.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def git(repo, *args):
    return subprocess.check_output(['git', '--no-optional-locks', '-C', str(repo), *args]).decode().strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--artifact-root', type=Path, required=True)
    a = p.parse_args()
    source, bundle = a.source.resolve(), a.artifact_root.resolve()
    assert source != ROOT and not bundle.is_relative_to(source)
    assert not bundle.exists(), 'Do not overwrite an existing artifact package'
    expected_commit = git(source, 'rev-parse', 'b8330a0')
    assert git(source, 'rev-parse', 'HEAD') == expected_commit
    assert not git(source, 'status', '--porcelain'), 'Source must stay frozen'
    rows = json.loads((source/'docs/w37/migration-audit-20260909/files.json').read_text())
    unique = {}
    for r in rows:
        key = r['resolved_path']
        if key in unique:
            assert unique[key]['sha256'] == r['sha256']
        unique[key] = r
    errors = []
    for path, r in unique.items():
        actual = sha(path) if Path(path).is_file() else None
        if actual != r['sha256'] or Path(path).stat().st_size != r['bytes']:
            errors.append(dict(path=path, expected=r['sha256'], actual=actual))
    if errors:
        print(json.dumps(errors, indent=2)); raise SystemExit('Frozen SHA mismatch; no import performed')
    assert len(unique) == 371
    bundle.mkdir(parents=True)
    mapping, objects = {}, []
    for path, r in unique.items():
        relative = 'files/' + path.lstrip('/')
        dest = bundle/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        assert sha(dest) == r['sha256']
        objects.append(dict(relative=relative, sha256=r['sha256'], size_bytes=r['bytes']))
    for r in rows:
        relative = 'files/' + r['resolved_path'].lstrip('/')
        mapping[r['path']] = relative
        mapping[r['resolved_path']] = relative
    # Source files in the audited set plus the local execution/report closure.
    selected = set()
    for r in rows:
        src = Path(r['path'])
        if src.is_relative_to(source):
            rel = src.relative_to(source)
            if rel.parts[0] != 'results':
                selected.add(str(rel))
    prefixes = ('research/w37/onef1b/', 'research/w37/maintenance/',
                'scripts/w37/', 'src/x10000_analysis/', 'workflow/scripts/')
    tracked = git(source, 'ls-files').splitlines()
    for rel in tracked:
        if rel.startswith(prefixes):
            selected.add(rel)
    for rel in ('workflow/Snakefile', 'workflow/config/case256.yaml',
                'workflow/config/dag_mfu_versions.toml', 'workflow/config/dag_mfu_w37_versions.toml',
                'workflow/DAG_MFU_VERSION_PIPELINE.md', 'docs/w37/coordination/code_manifest.json',
                'docs/w37/MFU_TMUX_V610_HANDOFF.md', 'docs/w37/MFU_MODEL_INTEGRATION_PLAN.md',
                'docs/w37/MFU_MIGRATION_AUDIT_20260909.md'):
        selected.add(rel)
    selected.update(str(f.relative_to(source)) for f in (source/'docs/w37/migration-audit-20260909').glob('*.json'))
    selected.update(str(f.relative_to(source)) for f in (source/'docs/w37/1f1b/v610').rglob('*') if f.is_file())
    selected.update(str(f.relative_to(source)) for f in (source/'docs/w37/1f1b/binding').rglob('*') if f.is_file())
    imported = []
    for rel in sorted(selected):
        src = source/rel
        assert src.is_file(), rel
        dest = ROOT/'vendor/fab_w37'/rel
        assert not dest.exists(), str(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        imported.append(dict(path=str(dest.relative_to(ROOT)), source_path=str(src), sha256=sha(src)))
    docs = ROOT/'docs/v610'
    put(docs/'artifact_manifest.json', dict(schema='mfu-v610-portable-artifacts-v1',
        source_commit=expected_commit, audit_commit=git(source,'rev-parse','146cc97'),
        historical_records=rows, objects=objects, logical_paths=mapping,
        raw_copied=False, raw_scanned=False, source_files_modified=False))
    put(docs/'imported_code.json', dict(source_commit=expected_commit, files=imported))
    put(ROOT/'.local/v610.json', dict(artifact_root=str(bundle), raw_root=None,
        source_snapshot=str(source), runs_root=str(ROOT/'results/v610')))
    put(docs/'import_acceptance.json', dict(status='PASS', unique_files=len(unique),
        bytes=sum(r['bytes'] for r in unique.values()), records=len(rows),
        copied_code_files=len(imported), source_sha_mismatches=0, raw_files_read=0))
    print(json.dumps(dict(status='PASS', files=len(unique), code_files=len(imported), bundle=str(bundle))))


if __name__ == '__main__':
    main()
