#!/usr/bin/env python3
"""Review frozen outputs, archived running code, active links, and exact regression."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'results/w37/A'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=BASE/'post685-final-verification');p.add_argument('--reproduction-id',default='round2-entry-check');a=p.parse_args();out=a.output.resolve()
    assert out.is_relative_to(BASE);out.mkdir(parents=True,exist_ok=False)
    checked=[];code=[]
    for version,run in [('v686','causal-r3'),('v687','readiness-r1'),('v688','scenarios-r2')]:
        base=BASE/f'post685-candidate-{run}';archive=ROOT/f'research/w37/onef1b/post685/history/{version}-research1'
        if version=='v688':archive=archive.with_name('v688-research1-r2')
        manifest=json.loads((base/'run_manifest.json').read_text());assert manifest['exit_code']==0
        for item in manifest['artifacts']:
            f=Path(item['path']);f=f if f.is_absolute() else base/f
            assert sha(f)==item['sha256'],f;checked.append(str(f))
        seal=json.loads((base/'prediction_seal.json').read_text())
        for item in seal['files']:assert sha(base/item['path'])==item['sha256']
        for item in json.loads((base/'command.json').read_text())['code']:
            f=Path(item['path']);match=f if sha(f)==item['sha256'] else archive/f.name
            assert match.exists() and sha(match)==item['sha256'],(version,f,match)
            code.append(dict(run=run,original=str(f),verified_copy=str(match),sha256=item['sha256']))
        audit=json.loads((base/'input_access_audit.json').read_text());assert not audit['raw_trace_scanned'];assert audit['denial_probes_passed']==3
        for read in audit['reads']:
            if 'evaluator' in '|'.join(read['roles']) or '/evaluator_only/' in read['path']:
                assert set(read['phases']).issubset({'hash_preflight','evaluator'}),read
        recon=pd.read_csv(base/'source_validation/observed_cost_reconstruction.csv');assert recon.max_start_error_ns.max()==0 and recon.max_end_error_ns.max()==0
        # Reproduction creates source candidates with the same model, even when
        # the active runner has gained a selector or new additive input pins.
        fresh=BASE/f'post685-candidate-{a.reproduction_id}-{version}'
        assert json.loads((fresh/'run_manifest.json').read_text())['exit_code']==0
        for rel in ['evaluator_only/iteration_results.csv','evaluator_only/phase_results.csv','source_validation/iteration_results.csv']:
            pd.testing.assert_frame_equal(pd.read_csv(base/rel),pd.read_csv(fresh/rel),check_exact=True)
    formal=json.loads((ROOT/'docs/w37/coordination/code_manifest.json').read_text())['files']
    for f in formal:assert sha(ROOT/f['path'])==f['sha256'],f['path']
    syntax=[]
    for directory in [ROOT/'research/w37/onef1b',ROOT/'tests/w37/onef1b',ROOT/'docs/w37/1f1b']:
        for f in directory.rglob('*.py'):
            if 'history' in f.parts:continue
            ast.parse(f.read_text());syntax.append(str(f))
    links=[]
    for f in [ROOT/'docs/w37/1f1b'/n for n in ['REPORT.md','HANDOFF.md','GOAL.md','post685/SEMANTICS.md']]:
        for target in re.findall(r'\]\(([^)]+)\)',f.read_text()):
            if target.startswith(('https://','http://','#')):continue
            path=target.split('#')[0];path=re.sub(r':\d+$','',path)
            dest=(f.parent/path).resolve();assert dest.exists(),(f,target);links.append(dict(document=str(f),target=target))
    diagnostic=BASE/'post685-candidate-scenarios-r2/evaluator_only'
    sh=pd.read_csv(diagnostic/'diagnostic_shapley_components.csv').groupby('iteration').attributed_delta_ms.sum()
    ld=pd.read_csv(diagnostic/'diagnostic_lane0_error_ledger.csv').set_index('iteration')
    assert (sh-ld.phase_component_effect_ms).abs().max()<1e-7
    assert (sh+ld.remaining_schedule_pp_initial_skew_error_ms-ld.total_underprediction_ms).abs().max()<1e-7
    m=pd.read_csv(BASE/'post685-delivery-round2-final/version_phase_metrics.csv')
    for ph in ['entry','tail','outer']:
        x=m[m.phase.eq(ph)&m.split.eq('development_primary')]
        assert x.predicted_mean_ms.nunique()==1 and x.mape_pct.nunique()==1
    result={'status':'PASS','artifact_files_verified':len(checked),'archived_code_and_manifest_checks':code,'frozen_formal_files_unchanged':len(formal),
        'python_syntax_checked':syntax,'active_document_links':links,'exact_reproduction_versions':['v686','v687','v688'],
        'source_observed_reassembly_error_ns':0,'target_guard_passed':True,'lane0_shapley_and_residual_conserved':True,
        'source_sibling_weekly_changes_performed':False,'decision':'retain v685; v687 PP mechanism useful, target runtime transfer unverified'}
    (out/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['archived_code_and_manifest_checks','python_syntax_checked','active_document_links']},indent=2))


if __name__=='__main__':main()
