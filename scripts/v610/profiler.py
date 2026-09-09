"""Profiler-first analysis boundary; no raw work without an explicit opt-in.

The default validates frozen source observations only. Optional inventories use
unchanged historical scripts, not a new profiler-only v610 cost method.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
from x10000_analysis.v610 import ROOT, VENDOR, mapped_spec, checked_input, checked_run, dump, sha


def inventory(kind, raw_root, out, rank_host_map=None):
    raw_root = raw_root.resolve(strict=True)
    if not raw_root.is_dir() or raw_root in (Path('/'), Path.home(), ROOT, VENDOR):
        raise ValueError('Specify one dataset directory, not a workspace or home root')
    out = checked_run(out)
    if out.is_relative_to(raw_root) or raw_root.is_relative_to(out):
        raise ValueError('Input and output trees must be disjoint')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Nonempty output: choose a new run directory')
    script = VENDOR/'case_256gpu_pp16_cp2_a2a/scripts'/('inventory_framework.py' if kind=='profiler' else 'inventory_hardware.py')
    argv = [sys.executable, '-B', str(script), '--root', str(raw_root), '--output-dir', str(out)]
    if kind == 'communication':
        if rank_host_map is None or not rank_host_map.is_file():
            raise ValueError('communication inventory requires an existing rank-host-map')
        argv += ['--rank-host-map', str(rank_host_map.resolve()), '--workers', '1']
    out.mkdir(parents=True, exist_ok=True)
    cmd = ['/usr/bin/bwrap', '--die-with-parent', '--ro-bind', '/', '/', '--dev','/dev',
           '--tmpfs','/tmp','--ro-bind',str(raw_root),str(raw_root),'--bind',str(out),str(out),'--unshare-net']
    if kind == 'communication':
        cmd += ['--ro-bind',str(rank_host_map.resolve()),str(rank_host_map.resolve())]
    cmd += ['--',*argv]
    dump(out/'command.json',dict(argv=cmd,script_sha256=sha(script),raw_root=str(raw_root),
         capability='file_identity_only' if kind=='profiler' else 'optional_hardware_csv_diagnostics',
         modifies_v610_parameters=False))
    with (out/'execution.log').open('w') as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=900,
                       env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['derived','profiler','communication'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--raw-root',type=Path)
    p.add_argument('--rank-host-map',type=Path)
    p.add_argument('--allow-raw-inventory',action='store_true')
    a=p.parse_args()
    if a.mode == 'derived':
        output=checked_run(a.output)
        if output.exists(): raise ValueError('Existing output is immutable: choose a new filename')
        record=checked_input(mapped_spec('binding')['inputs']['observations'])
        dump(output,dict(status='PASS',source_profiler_observations=record,raw_trace_reads=0,
                         communication_logs_required=False,raw_to_all_costs_closed=False))
    else:
        if not a.allow_raw_inventory or a.raw_root is None:
            p.error('Raw inventory requires --allow-raw-inventory AND an explicit --raw-root')
        inventory(a.mode,a.raw_root,a.output,a.rank_host_map)


if __name__=='__main__': main()
