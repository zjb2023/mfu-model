"""Archive small acceptance evidence; retain large result files outside Git."""
from __future__ import annotations
import json
import platform
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
from x10000_analysis.v610 import ROOT, sha, dump, artifact_root, STAGES

def git(path, *args):
    return subprocess.check_output(['git','--no-optional-locks','-C',str(path),*args]).decode().strip()


def main():
    run=ROOT/'results/v610'
    out=ROOT/'docs/v610/acceptance'
    assert not out.exists(), 'Acceptance archive is immutable; do not overwrite'
    verification=json.loads((run/'verification-final.json').read_text())
    assert verification['status']=='PASS'
    release_browser=json.loads((run/'webcheck-release/acceptance.json').read_text())
    history_browser=json.loads((run/'webcheck-history/inline_acceptance.json').read_text())
    assert release_browser['status']=='PASS'
    assert len(history_browser)==23 and all(c['passed'] for c in history_browser)
    suites=ET.parse(run/'tests-final.xml').getroot()
    test_counts={key:sum(int(s.attrib.get(key,0)) for s in suites.iter('testsuite'))
                 for key in ['tests','failures','errors','skipped']}
    assert test_counts == dict(tests=32,failures=0,errors=0,skipped=0)
    for name in ['release-final.dryrun.log','binding-final.dryrun.log']:
        text=(run/name).read_text()
        assert 'Nothing to be done' in text and 'missing metadata' not in text, name
    copies={
        'verification-final.json':'verification.json','tests-final.xml':'tests.xml',
        'webcheck-release/acceptance.json':'browser-release.json',
        'webcheck-history/inline_acceptance.json':'browser-history.json',
        'webcheck-history/inline_link_audit.json':'browser-history-links.json',
        'release-final/model/release.json':'release.json',
        'release-final/model/predictions.csv':'predictions.csv',
        'release-final/evaluate/iteration_results.csv':'iteration_results.csv',
        'release-final/evaluate/metrics.json':'metrics.json',
        'release-final/evaluate/phase_iteration_results.csv':'phase_iteration_results.csv',
        'release-final/render/view_checks.json':'view_checks.json',
        'binding-final/audit/summary.json':'binding-audit.json',
        'binding-final/review/review.json':'binding-review.json',
        'binding-final/model/predictions.csv':'binding-predictions.csv',
        'binding-final/evaluate/metrics.csv':'binding-metrics.csv',
        'binding-final/evaluate/iteration_results.csv':'binding-iteration-results.csv',
        'profiler-input-check/derived_inputs.json':'profiler-derived-inputs.json',
        'release-final.dryrun.log':'release-dryrun.log',
        'binding-final.dryrun.log':'binding-dryrun.log',
    }
    out.mkdir(parents=True)
    artifacts=[]
    for src,dst in copies.items():
        shutil.copy2(run/src,out/dst)
        artifacts.append(dict(path=str(out/dst),relative_to_repo=str((out/dst).relative_to(ROOT)),
                              source_path=str(run/src),sha256=sha(out/dst),size_bytes=(out/dst).stat().st_size))
    stages=[]
    for pipeline,folder in [('release','release-final'),('binding','binding-final')]:
        for stage in STAGES[pipeline]:
            directory=run/folder/stage
            completion=json.loads((directory/'complete.json').read_text())
            files=[]
            for relative,expected in completion['files'].items():
                path=directory/relative
                assert sha(path)==expected, str(path)
                files.append(dict(path=str(path),sha256=expected,size_bytes=path.stat().st_size))
            stages.append(dict(pipeline=pipeline,stage=stage,status='PASS',elapsed_s=completion['elapsed_s'],
                          complete_json=str(directory/'complete.json'),complete_sha256=sha(directory/'complete.json'),
                          inputs=str(directory/'resolved_inputs.json'),config=str(directory/'execution_config.json'),
                          command=str(directory/'command.json'),log=str(directory/'execution.log'),files=files))
    providers={}
    for name,path in [('original_mfu','/home/zjb/Desktop/mfu-model'),
                      ('original_224','/home/zjb/Desktop/mfu-model-parallel-strategy-224gpu'),
                      ('w37_source','/home/zjb/Desktop/worktrees/fab-w37-1f1b')]:
        providers[name]=dict(path=path,head=git(path,'rev-parse','HEAD'),branch=git(path,'branch','--show-current'),
                            status=git(path,'status','--short'),staged_diff=git(path,'diff','--cached','--stat'))
    assert providers['w37_source']['status']=='' and providers['w37_source']['head'].startswith('b8330a0')
    assert providers['original_mfu']['head'].startswith('e71ccb0') and providers['original_mfu']['staged_diff']==''
    code=[]
    for rel in git(ROOT,'ls-files','src/x10000_analysis/v610.py','scripts/v610','workflow','configs/v610',
                   'requirements-v610.txt','requirements-v610.lock.txt','package.json','package-lock.json').splitlines():
        code.append(dict(path=str(ROOT/rel),sha256=sha(ROOT/rel)))
    dump(ROOT/'docs/v610/run_manifest.json',dict(schema='mfu-v610-execution-manifest-v1',status='PASS',
        worktree=str(ROOT),branch=git(ROOT,'branch','--show-current'),code_commit=git(ROOT,'rev-parse','HEAD'),
        provider_commit=verification['source_commit'],provider_audit_commit=verification['audit_commit'],
        artifact_root=str(artifact_root()),artifact_manifest=str(ROOT/'docs/v610/artifact_manifest.json'),
        artifact_manifest_sha256=sha(ROOT/'docs/v610/artifact_manifest.json'),
        imported_code_manifest=str(ROOT/'docs/v610/imported_code.json'),
        provider_git_provenance=str(ROOT/'docs/v610/provider_git_provenance.json'),
        code=code,stages=stages,evidence=artifacts,tests=test_counts,providers_after=providers,
        python=platform.python_version(),platform=platform.platform(),
        system_bwrap_sha256=sha('/usr/bin/bwrap'),service_snapshot=json.loads((run/'service.json').read_text()),
        raw_reads=0,raw_moved=False,raw_in_git=False,
        reproduction_scope='sealed v610 and original source-derived binding, not new accuracy research'))
    print(json.dumps(dict(status='PASS',evidence_files=len(artifacts),stages=len(stages),tests=test_counts)))


if __name__=='__main__':main()
