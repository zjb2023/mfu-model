"""One-time provenance audit: distinguish provider Git blobs from SHA-only history."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from import_snapshot import ROOT, put, sha

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);a=p.parse_args()
def git(*args):
    return subprocess.check_output(['git','--no-optional-locks','-C',str(a.source),*args])
commit=git('rev-parse','HEAD').decode().strip()
assert commit == 'b8330a0ffc1c4d73c0a2403291b89edd55999a58'
tree={}
for line in git('ls-tree','-rz','HEAD').split(b'\0'):
    if line:
        header, name=line.split(b'\t',1)
        tree[name.decode()]=header.decode().split()[2]
records=[]
for r in json.loads((ROOT/'docs/v610/imported_code.json').read_text())['files']:
    f=ROOT/r['path'];rel=r['path'].removeprefix('vendor/fab_w37/')
    assert sha(f)==r['sha256']
    blob=hashlib.sha1(b'blob '+str(f.stat().st_size).encode()+b'\0'+f.read_bytes()).hexdigest()
    if rel in tree: assert blob==tree[rel], 'Provider tracked file differs from commit: '+rel
    records.append(dict(path=r['path'],sha256=r['sha256'],provider_git_blob=tree.get(rel),
                        identity='provider_commit_blob' if rel in tree else 'prior_audit_sha_only_evidence'))
put(ROOT/'docs/v610/provider_git_provenance.json',dict(status='PASS',provider_commit=commit,files=records,
    tracked_count=sum(r['provider_git_blob'] is not None for r in records),
    sha_only_count=sum(r['provider_git_blob'] is None for r in records)))
