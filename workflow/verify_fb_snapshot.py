"""Offline frozen-template verification: never opens raw traces or old worktree."""
import argparse,gc,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'workflow/data_foundation'))
from assemble_pp32_detailed_fb_r1 import compose as compose32
from extrapolate_detailed_fb_32to256_r1 import compose as compose256
def digest(x):return hashlib.sha256((json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();assert not a.out.exists()
    base=ROOT/'results/data-foundation';snapshot=json.loads((ROOT/'docs/data-foundation/FB_BRANCH_SNAPSHOT.json').read_text())
    for f in snapshot['files']:
        p=ROOT/f['relative_path'];assert p.is_file() and not p.is_symlink();assert hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256'],str(p)
    templates={ph:json.loads((base/name/'graph.json').read_text()) for ph,name in [('F','four-layer-f-r1'),('B','b-four-layers-r10')]}
    params=json.loads((base/'pp32-detailed-fb-assembly-r1/parameters.json').read_text());results=[]
    for label in ['pp32-detailed-fb-assembly-r1','detailed-fb-32to256-r1']:
        if label.startswith('pp32'):g,t,_=compose32(params,templates);end='s0:B7'
        else:g,t,_=compose256(16,4,params,templates);end='s0:B3'
        expected=json.loads((base/label/'sealed-prediction.json').read_text())['window_ms'];actual=t[end]['end_ms']-t['s0:F0']['start_ms'];assert abs(actual-expected)<1e-6
        for filename,obj in [('graph.json',g),('replay.json',t)]:
            old=next(x for x in snapshot['external_expanded_artifacts'] if x['path'].endswith('/'+label+'/'+filename));assert digest(obj)==old['sha256'],(label,filename)
        results.append(dict(model=label,window_ms=actual,expanded_nodes=len(g),graph_and_replay_sha256='MATCH_FROZEN_ORIGINAL'))
        del g,t;gc.collect()
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(dict(status='PASS',snapshot_files_verified=len(snapshot['files']),models=results,raw_trace_read=False,old_worktree_read=False),indent=2)+'\n');print(a.out.read_text())
if __name__=='__main__':main()
