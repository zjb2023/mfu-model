"""Read-only cleanup inventory. Never deletes or modifies inspected files; no symlink traversal."""
from pathlib import Path
import collections,csv,gzip,hashlib,json,os,subprocess,time
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/w37/cleanup-audit-20260909';OUT.mkdir(parents=True,exist_ok=True)
tracked=set(subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0'))
files=[];links=[];allocated=0;groups=collections.defaultdict(list);top=collections.Counter();cache=[]
for dp,ds,fs in os.walk(ROOT,followlinks=False):
    if Path(dp)==OUT:ds[:]=[];continue
    for name in list(ds)+fs:
        p=Path(dp)/name
        if p.is_symlink():links.append({'path':str(p.relative_to(ROOT)),'target':os.readlink(p),'resolved':str(p.resolve())})
    for name in fs:
        p=Path(dp)/name
        if p.is_symlink():continue
        st=p.stat();rel=str(p.relative_to(ROOT));row={'path':rel,'bytes':st.st_size,'allocated_bytes':st.st_blocks*512,'tracked':rel in tracked};files.append(row);allocated+=row['allocated_bytes']
        if rel.startswith('results/w37/A/'):
            top[rel.split('/')[3]]+=row['allocated_bytes']
            if st.st_size:groups[st.st_size].append(row)
        parts=Path(rel).parts
        if not row['tracked'] and (any(x in parts for x in ['__pycache__','.pytest_cache']) or ('.snakemake' in parts and 'iocache' in parts)):
            cache.append(row)
print('Inventory complete; hashing size-matched derived result files for duplicates.',flush=True)
duplicates=[];hashed_bytes=0;deadline=time.monotonic()+30;hash_budget=64*1024*1024;hashed_files=0
for size,rows in sorted(groups.items(),reverse=True):
    if size<1024*1024 or len(rows)<2:continue
    if time.monotonic()>deadline or hashed_bytes+size>hash_budget:break
    hashes=collections.defaultdict(list)
    for row in rows:
        if time.monotonic()>deadline or hashed_bytes+size>hash_budget:break
        h=hashlib.sha256()
        with (ROOT/row['path']).open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
        hashes[h.hexdigest()].append(row['path']);hashed_bytes+=size;hashed_files+=1
    for digest,paths in hashes.items():
        if len(paths)>1:duplicates.append({'sha256':digest,'bytes_each':size,'copies':len(paths),'duplicate_logical_bytes':size*(len(paths)-1),'paths':paths})
duplicates.sort(key=lambda d:d['duplicate_logical_bytes'],reverse=True)
with gzip.open(OUT/'files.csv.gz','wt',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(files[0]));w.writeheader();w.writerows(files)
protected={}
for link in links:
    try:rel=Path(link['resolved']).relative_to(ROOT/'results/w37/A')
    except ValueError:continue
    if rel.parts:protected.setdefault(rel.parts[0],[]).append(link['path'])
summary={'scope':str(ROOT),'files':len(files),'logical_bytes':sum(r['bytes'] for r in files),'allocated_bytes':allocated,'tracked_logical_bytes':sum(r['bytes'] for r in files if r['tracked']),'cache_candidate_allocated_bytes':sum(r['allocated_bytes'] for r in cache),'cache_candidate_logical_bytes':sum(r['bytes'] for r in cache),'cache_candidate_files':len(cache),'duplicate_groups':len(duplicates),'duplicate_logical_bytes':sum(d['duplicate_logical_bytes'] for d in duplicates),'hashed_derived_bytes':hashed_bytes,'hashed_files':hashed_files,'duplicate_check_scope':'Sample only: size-matched derived files >=1MiB, 64MiB byte cap and 30 second soft deadline; not exhaustive','largest_result_directories':[{'path':'results/w37/A/'+p,'allocated_bytes':n,'symlink_references':protected.get(p,[])} for p,n in top.most_common(20)],'notes':['Inspection only; no files deleted.','Exact duplicate bytes do not imply removable paths: manifests, seals and historical reproduction may depend on each path.','Cache candidate list excludes tracked files and all symlinks. Snakemake metadata/logs and code snapshots are not classified as disposable.','Do not follow symlinks for deletion; original project and sibling worktrees excluded.','Memory status is separate from disk usage.']}
for name,obj in [('summary.json',summary),('cache_candidates.json',cache),('duplicate_groups.json',duplicates),('symlinks.json',links)]:
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:summary[k] for k in ['allocated_bytes','logical_bytes','cache_candidate_allocated_bytes','cache_candidate_files','duplicate_groups','duplicate_logical_bytes','hashed_derived_bytes']},indent=2),flush=True)
