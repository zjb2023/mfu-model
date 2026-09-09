"""Stage input capabilities and checked immutable dependency manifests."""
import json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'scripts/w37'))
from smoke_worker import InputGuard,sha


class StageGuard(InputGuard):
    def __init__(self,out,items,stage,allowed_results,raw_parse_paths=(),hash_only_results=(),target_raw_parse_paths=()):
        super().__init__('A',out,items,'snakemake_stage');self.stage=stage;self.allowed_results=allowed_results
        self.raw_parse_paths={str(Path(p).resolve()) for p in raw_parse_paths}
        self.hash_only_results=[Path(p).resolve() for p in hash_only_results]
        self.target_raw_parse_paths={str(Path(p).resolve()) for p in target_raw_parse_paths}
        self.sealed_reference_verified=False
    def event(self,event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            path=Path(os.fsdecode(args[0])).resolve();item=self.items.get(str(path))
            writing=bool(args[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if not writing:
                target=item and ('evaluator' in '|'.join(item['roles']) or '/evaluator_only/' in item['path'])
                if target and self.phase not in ['hash_preflight','evaluator']:raise PermissionError('target timing is evaluator-only after independent seal')
                if item and 'evaluator_cached_target_runtime_NEVER_MODEL_FIT' in item['roles'] and self.phase!='hash_preflight':
                    if not (self.stage=='diagnose' and self.phase=='evaluator' and self.sealed_reference_verified):
                        raise PermissionError('cached target runtime requires sealed diagnose/evaluator and is never a model input')
                if item and item.get('raw_trace') and self.phase!='hash_preflight':
                    source_ok=self.stage=='diagnose' and self.phase=='source_only' and not target and str(path) in self.raw_parse_paths
                    target_ok=self.stage=='diagnose' and self.phase=='evaluator' and target and self.sealed_reference_verified and str(path) in self.target_raw_parse_paths
                    if not (source_ok or target_ok):
                        raise PermissionError('raw parsing requires explicit bounded source intake or sealed target posthoc intake; default stages may hash only')
                if path.is_relative_to(ROOT/'results/w37/A') and not path.is_relative_to(self.output):
                    hash_admitted=self.phase=='hash_preflight' and any(path.is_relative_to(p) for p in self.hash_only_results)
                    if not hash_admitted and not any(path.is_relative_to(p) for p in self.allowed_results):raise PermissionError('undeclared result input (including past target diagnostics)')
        super().event(event,args)


def checked_stage(path):
    marker=json.loads((path/'complete.json').read_text());assert marker['status']=='PASS'
    assert sha(path/'run_manifest.json')==marker['manifest_sha256']
    for item in json.loads((path/'run_manifest.json').read_text())['artifacts']:assert sha(path/item['path'])==item['sha256']
