"""Standalone F/B ordinary-AlltoAll structure + eight-rank measured timelines."""
from pathlib import Path
import hashlib,shutil,urllib.request,orjson

def load(p):return orjson.loads(p.read_bytes())
def save(p,value):p.write_bytes(orjson.dumps(value,option=orjson.OPT_INDENT_2))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation';OUT=BASE/'mccl-fb-structure-r1'
SERVE=Path('/home/zjb/Desktop/worktrees/mfu-w37-v610/results/data-foundation/2111-blocks-ui-r1/df-v001')

def main():
    OUT.mkdir(exist_ok=True);inputs={};payload={};count=0
    def read(p):inputs[str(p)]=sha(p);return load(p)
    def pack(d,label,typ):
        n=d.get('input_nelems') or 0;dtype=d.get('dtype') or '';size={'BFloat16':2,'Float':4,'Long':8}.get(dtype,0)
        return dict(s=d['start_ms'],e=d['end_ms'],id=d['event'],label=label,type=typ,collective=d.get('collective'),
                    group=d.get('group'),nr=len(d.get('ranks') or []),nelems=n,dtype=dtype,mib=n*size/1048576)
    for it in range(155,201,5):
        payload[it]={ph:{u:[] for u in range(4)} for ph in ['F','B']}
        for r in range(8,16):
            d=read(BASE/f'full155-device-inventory-r1/i{it}-r{r}.json');ev=read(BASE/f'full155-cp-attention-r1/evidence-i{it}-r{r}.json')
            for ph in ['F','B']:
                for u in range(4):
                    ds=[x for x in d['phases'][ph]['devices'] if x['unit']==u]
                    eps=sorted((x for x in ds if x.get('group')=='EXPERT_MODEL_PARALLEL_GROUP' and x.get('collective')),key=lambda x:x['start_ms'])
                    assert len(eps)==(3 if ph=='F' else 6)
                    regions={}
                    for direction,start in ([('forward',0)] if ph=='F' else [('recompute',0),('backward',3)]):
                        labels=['token dispatch','概率/权重 dispatch','专家输出 combine'] if direction!='backward' else ['输出梯度送向专家','概率/权重梯度返回','token梯度返回']
                        events=[pack(eps[start+j],labels[j],'EP') for j in range(3)]
                        prefix='expert_' if direction=='forward' else 'expert_'+direction+'_'
                        for x in ds:
                            if x['kind'].startswith(prefix):
                                part=x['kind'][len(prefix):]
                                if part in ['FC1','FC2','activation']:events.append(pack(x,('梯度 ' if direction=='backward' else '')+part,part))
                        if direction!='backward':
                            events.extend(pack(x,'计数/元数据','metadata') for x in ds if x['kind'] in ['EP_metadata','other_communication'] and x.get('collective'))
                        regions[direction+'_ep']=sorted(events,key=lambda x:x['s'])
                        s=next(s for s in ev['scopes'] if s['phase']==ph and s['unit']==u and s['direction']==direction)
                        regions[direction+'_cp']=[pack(x,f'CP{j}','CP') for j,x in enumerate(s['cp'])]+[pack(x,'Flash '+direction,'Flash') for x in s['flash']]
                        assert len(s['cp'])==(5 if direction=='backward' else 4)
                    payload[it][ph][u].append(dict(rank=r,regions=regions,start_ms=min(x['start_ms'] for x in ds),end_ms=max(x['end_ms'] for x in ds)));count+=1
    template=Path(__file__).with_name('ui_mccl_fb_r1.html');inputs[str(template)]=sha(template);inputs[str(Path(__file__))]=sha(Path(__file__))
    payload=orjson.loads(orjson.dumps(payload,option=orjson.OPT_NON_STR_KEYS))
    save(OUT/'payload.json',payload)
    legacy={'60':{ph:{str(u):[] for u in range(4)} for ph in ['F','B']}}
    for r in range(8,16):
        old=read(BASE/f'full155-backend-diagnosis-r1/old-r{r}.json')
        expert={ph:read(BASE/f'{folder}/measurements/old-i60-r{r}.json') for ph,folder in [('F','stable-four-layer-f-r1'),('B','stable-b-compute-r1')]}
        assert all(x['sha256']==old['sha256'] for x in expert.values())
        for ph in ['F','B']:
            for u in range(4):
                ds=[x for x in old['phases'][ph]['devices'] if x['unit']==u]
                cp=sorted((x for x in ds if x.get('group')=='CONTEXT_PARALLEL_GROUP' and x.get('collective')),key=lambda x:x['start_ms'])
                assert len(cp)==(4 if ph=='F' else 9)
                deep=[x for x in ds if 'deep_ep::' in x['name']]
                parts=expert[ph]['layers'][u]['devices'];regions={}
                if ph=='B':
                    # Distinguish the backward permutation from forward/recompute's true specialization.
                    starts=[x['start_ms'] for x in deep if 'moe_permute_mask_kernel' in x['name'] and ', false>' in x['name']]
                    assert len(starts)==1,(r,u,starts)
                    split=starts[0]
                else:split=float('inf')
                for direction in (['forward'] if ph=='F' else ['recompute','backward']):
                    selected=deep if ph=='F' else [x for x in deep if (x['start_ms']<split)==(direction=='recompute')]
                    events=[pack(x,'DeepEP可见活动：'+x['name'],'DeepEP') for x in selected]
                    prefix='' if ph=='F' else direction+'_'
                    for x in parts:
                        if x['part'].startswith(prefix):
                            part=x['part'][len(prefix):]
                            if part in ['FC1','FC2','activation']:events.append(pack(x,('梯度 ' if direction=='backward' else '')+part,part))
                    regions[direction+'_ep']=sorted(events,key=lambda x:x['s'])
                    cs=cp if ph=='F' else (cp[:4] if direction=='recompute' else cp[4:])
                    flashes=[x for x in ds if 'flash_atten' in x['name'] and (('_bwd_' in x['name'])==(direction=='backward'))]
                    assert len(flashes)==1,(r,ph,u,direction)
                    regions[direction+'_cp']=[pack(x,f'CP{j}（旧源）','CP') for j,x in enumerate(cs)]+[pack(x,'Flash '+direction,'Flash') for x in flashes]
                    # These cached old records have no message dimensions; do not invent them.
                    for e in regions[direction+'_cp']:
                        if e['collective']:e.update(nr=2,nelems=None,dtype='未记录',mib=None)
                legacy['60'][ph][str(u)].append(dict(rank=r,regions=regions,start_ms=min(x['start_ms'] for x in ds),end_ms=max(x['end_ms'] for x in ds)))
    save(OUT/'legacy-payload.json',legacy)
    # Inline data makes the exported HTML usable offline as well.
    text=template.read_text().replace('__PAYLOAD__',orjson.dumps(payload).decode().replace('</','<\\/')).replace('__LEGACY_PAYLOAD__',orjson.dumps(legacy).decode().replace('</','<\\/'))
    (OUT/'index.html').write_text(text)
    save(OUT/'manifest.json',dict(inputs=inputs,new_views=count,legacy_views=64,source_traces=88,new_source_traces=80,legacy_source_traces=8,scope='PP1 F0/B0 four layers EP8; new155-200 plus old60; conditional structure, measured selected events; old DeepEP transport visibility incomplete',
                                 outputs={str(OUT/n):sha(OUT/n) for n in ['index.html','payload.json','legacy-payload.json']}))
    dest=SERVE/'mccl-fb-structure-r1';dest.mkdir(exist_ok=True)
    for n in ['index.html','payload.json','legacy-payload.json','manifest.json']:shutil.copyfile(OUT/n,dest/n)
    url='http://192.168.8.16:43312/df-v001/mccl-fb-structure-r1/'
    assert urllib.request.urlopen(url,timeout=20).read()==(OUT/'index.html').read_bytes()
    save(OUT/'verification.json',dict(http='PASS',views=count,url=url,browser_visual='NOT_TESTED'))
    print('PASS',url,'rank-phase-layer views',count)

if __name__=='__main__':main()
