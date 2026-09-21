"""Freeze current served overview and its local indexed page closure into Git tree."""
import hashlib,json,re,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
WEB=Path('results/data-foundation/2111-blocks-ui-r1/df-v001')
LIVE=Path('/home/zjb/Desktop/worktrees/mfu-w37-v610')/WEB
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    for name,files in {
        'prediction-overview-r2':['index.html','validation.json','data.json','b-only-diagnosis.json','sealed-prediction.json','diagnostic-report.json','report.json','display-policy.json','manifest.json','seal.json'],
        'b-position-curves-r1':['index.html','data.json','raw-provenance.json','manifest.json']}.items():
        dest=ROOT/WEB/name;dest.mkdir(parents=True,exist_ok=True)
        for file in files:
            if (dest/file).exists():assert sha(dest/file)==sha(LIVE/name/file)
            else:shutil.copyfile(LIVE/name/file,dest/file)
    # Follow references in copied HTML, including iframe and JS relative URLs.
    checked=set()
    while True:
        todo=[p for p in (ROOT/WEB).rglob('*.html') if p not in checked]
        if not todo:break
        for p in todo:
            checked.add(p);text=p.read_text()
            refs=set(re.findall(r'(?:href|src|data-src)=["\']([^"\']+)',text))
            refs.update(re.findall(r'["\'](\.\./[^"\'\s]+)["\']',text))
            for ref in refs:
                ref=ref.split('#')[0].split('?')[0]
                if not ref or ':' in ref or ref.startswith('/'):continue
                target=(p.parent/ref).resolve()
                assert target.is_relative_to(ROOT/WEB),target
                if target.exists():continue
                live_target=LIVE/target.relative_to(ROOT/WEB)
                assert live_target.exists(),(p,ref)
                members=list(live_target.rglob('*')) if live_target.is_dir() else [live_target]
                assert sum(q.stat().st_size for q in members if q.is_file())<100_000_000
                assert all(not q.name.endswith('.pt.trace.json') for q in members)
                target.parent.mkdir(parents=True,exist_ok=True)
                if live_target.is_dir():shutil.copytree(live_target,target)
                else:shutil.copyfile(live_target,target)
    files=[];links=[]
    for p in (ROOT/WEB).rglob('*'):
        if not p.is_file():continue
        assert sha(p)==sha(LIVE/p.relative_to(ROOT/WEB)),p
        files.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size))
        if p.suffix=='.html':
            text=p.read_text()
            refs=set(re.findall(r'(?:href|src|data-src)=["\']([^"\']+)',text))
            refs.update(re.findall(r'["\'](\.\./[^"\'\s]+)["\']',text))
            for ref in refs:
                ref=ref.split('#')[0].split('?')[0]
                if not ref or ':' in ref or ref.startswith('/'):continue
                target=(p.parent/ref).resolve()
                if target.is_dir():target=target/'index.html'
                assert target.exists(),(p,ref)
                assert target.is_relative_to(ROOT),target
                links.append(dict(page=str(p.relative_to(ROOT)),link=ref,target=str(target.relative_to(ROOT))))
    dest=ROOT/'docs/data-foundation/ERROR_ANALYSIS_RELEASE_R1.json'
    dest.write_text(json.dumps(dict(version='32to256-error-analysis-r1',entry=str(WEB/'prediction-overview-r2/index.html'),files=files,checked_links=links,conclusion='iter60 middle B mismatch explains87.8575% of1F1B overestimate; all B91.3545%; physical causes and held-out predictive accuracy not closed',raw_data_included=False),ensure_ascii=False,indent=2)+'\n')
    print('PASS',len(files),'files',len(links),'links')
if __name__=='__main__':main()
