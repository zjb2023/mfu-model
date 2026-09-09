"""Boundary tests use small fixtures, never raw or model evaluation data."""
import json
from pathlib import Path

import pytest

from x10000_analysis.v610 import checked_input, checked_run, resolve_artifact, sha, ROOT


def test_frozen_hash_change_fails_with_precise_identity(tmp_path):
    data = tmp_path/'input.csv'; data.write_text('value\n1\n')
    record = dict(path=str(data), sha256=sha(data), size_bytes=data.stat().st_size,
                  role='source_calibration')
    assert checked_input(record)['sha256'] == record['sha256']
    data.write_text('value\n2\n')
    with pytest.raises(ValueError, match='Frozen SHA mismatch:.*expected=.*actual='):
        checked_input(record)


def test_relocation_keeps_old_logical_name_and_hash(tmp_path):
    data = tmp_path/'new-location.csv'; data.write_text('value\n1\n')
    old = tmp_path/'old-location.csv'; old.symlink_to(data)
    record = dict(path=str(old), logical_path='dataset:source85',
                  sha256=sha(data), size_bytes=data.stat().st_size, role='source_calibration')
    result = checked_input(record)
    assert result['logical_path'] == 'dataset:source85'
    assert result['resolved_path'] == str(data)
    old.unlink(); old.symlink_to(tmp_path/'missing.csv')
    with pytest.raises(ValueError, match='Frozen SHA mismatch'):
        checked_input(record)


def test_output_cannot_escape_to_old_or_sibling_projects():
    for path in ['/home/zjb/Desktop/mfu-model/results', '/tmp/v610',
                 str(ROOT/'results/v610/../../docs')]:
        with pytest.raises(ValueError, match='Outputs must'):
            checked_run(path)
    assert checked_run(ROOT/'results/v610/test') == ROOT/'results/v610/test'


def test_unknown_input_cannot_be_silently_rebound(tmp_path):
    with pytest.raises(ValueError, match='Unregistered frozen input'):
        resolve_artifact('/raw/trace.json', {'logical_paths': {}}, tmp_path)


def test_frozen_mfu_constants_reconstruct_release():
    physics=json.loads((ROOT/'configs/v610/physics.json').read_text())
    mfu=100*physics['effective_flops_per_iteration']/(physics['world_size']*
        physics['peak_flops_per_gpu_s']*(22930.003646124995/1000))
    assert mfu == pytest.approx(3.285055047379364,abs=1e-13)


def test_profiler_inventory_without_communication_logs(tmp_path, monkeypatch):
    """Synthetic filenames only: original inventory must not parse trace bodies."""
    import importlib.util
    script=ROOT/'scripts/v610/profiler.py'
    spec=importlib.util.spec_from_file_location('profiler_adapter_test',script)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    raw=tmp_path/'synthetic'
    for rank in range(256):
        file=raw/f'run/worker{rank//8}/profiler/iteration_85/rank{rank}.123.pt.trace.json'
        file.parent.mkdir(parents=True,exist_ok=True)
        file.write_text('not JSON; filename-only fixture')
    output=tmp_path/'output'
    # Authorize only this test's temporary output; exercise the actual sandbox.
    def test_output_guard(path):
        assert path == output
        return path.resolve()
    monkeypatch.setattr(mod, 'checked_run', test_output_guard)
    mod.inventory('profiler',raw,output)
    report=json.loads((output/'framework_readiness.json').read_text())
    assert report['status']=='PASS_FILE_GRID' and report['profiler']['rank_count']==256
    assert report['category_counts']=={'profiler_trace':256}
    assert report['capability']=='file_identity_only'


def test_profiler_raw_inventory_needs_explicit_opt_in(tmp_path):
    import subprocess
    import sys
    result=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/v610/profiler.py'),
                          'profiler','--output',str(tmp_path/'out')],capture_output=True,text=True)
    assert result.returncode==2 and '--allow-raw-inventory' in result.stderr
    assert not (tmp_path/'out').exists()
