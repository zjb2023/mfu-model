"""Compare every selected B framework module without pooling microbatches."""
import argparse, csv, hashlib, json
from pathlib import Path
from audit_b_operator_position_r1 import union
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/data-foundation'
LABELS={
 'recompute_attention_CP':'重计算 Attention＋CP',
 'backward_attention_CP':'反向 Attention＋CP',
 'recompute_EP_dispatch':'重计算 EP分发',
 'recompute_EP_combine':'重计算 EP合并',
 'backward_EP_dispatch':'反向 EP分发',
 'backward_EP_combine':'反向 EP合并',
 'recompute_expert_linear_FC1':'重计算 专家FC1',
 'recompute_expert_linear_FC2':'重计算 专家FC2',
 'backward_expert_linear_FC1':'反向 专家FC1',
 'backward_expert_linear_FC2':'反向 专家FC2'}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 a.out.mkdir(parents=True,exist_ok=False);rows=[];coverage=[];inputs=[]
 for world,stages in [(224,[5,6]),(256,[1,14])]:
  ds=[]
  for stage in stages:
   p=BASE/f'b-framework-position-r1/{world}-pp{stage}.json';q=BASE/f'b-operator-position-r1/{world}-pp{stage}.json';inputs.extend([p,q])
   d=json.loads(p.read_text());raw=json.loads(q.read_text());ds.append(d)
   for block,original in zip(d['blocks'],raw['blocks']):
    selected=union([iv for op in block['operators'] for iv in op['intervals']])
    assert selected<=original['owned_device_union_ms']+1e-5
    coverage.append(dict(world=world,stage=stage,mb=block['mb'],B_ms=original['envelope_ms'],selected_union_ms=selected,
       other_device_only_ms=original['owned_device_union_ms']-selected,
       no_owned_device_ms=original['no_owned_device_coverage_ms']))
  for left,right in zip(ds[0]['blocks'],ds[1]['blocks']):
   l={x['kind']:x for x in left['summary']};r={x['kind']:x for x in right['summary']}
   for kind,label in LABELS.items():
    assert l[kind]['count']==r[kind]['count']==4
    rows.append(dict(world=world,mb=left['mb'],left_stage=stages[0],right_stage=stages[1],module=label,
      left_cpu_ms=l[kind]['cpu_sum_ms'],right_cpu_ms=r[kind]['cpu_sum_ms'],cpu_delta_ms=r[kind]['cpu_sum_ms']-l[kind]['cpu_sum_ms'],
      left_gpu_ms=l[kind]['gpu_union_ms'],right_gpu_ms=r[kind]['gpu_union_ms'],gpu_delta_ms=r[kind]['gpu_union_ms']-l[kind]['gpu_union_ms']))
 assert len(rows)==70 and len(coverage)==14
 with (a.out/'modules.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (a.out/'coverage.json').write_text(json.dumps(coverage,ensure_ascii=False,indent=2)+'\n')
 md=['# B框架模块全项对照\n','仅iter60四个代表rank；每行汇总同一B内4层的同类调用，不合并不同microbatch。差值为后stage减前stage，负值表示后stage更短。CPU调用与GPU设备活动分开，不是纯计算/纯通信成本。FC1/FC2按配对顺序推定。\n']
 for world,stages in [(224,[5,6]),(256,[1,14])]:
  for mb in range(3 if world==224 else 4):
   md.extend([f'## {world}卡 PP{stages[0]}→PP{stages[1]} · B{mb}\n', '| 模块 | 前CPU ms | 后CPU ms | CPU差值 | 前GPU ms | 后GPU ms | GPU差值 |','|---|---:|---:|---:|---:|---:|---:|'])
   for x in rows:
    if (x['world'],x['mb'])!=(world,mb):continue
    md.append('| '+x['module']+' | '+' | '.join(f'{x[k]:.3f}' for k in ['left_cpu_ms','right_cpu_ms','cpu_delta_ms','left_gpu_ms','right_gpu_ms','gpu_delta_ms'])+' |')
   md.append('')
 md+=['## 覆盖范围\n','| 卡数 | stage | B | B包络ms | 已选模块GPU并集ms | 其他设备独占覆盖ms | 无归属设备覆盖ms |','|---|---:|---:|---:|---:|---:|---:|']
 for c in coverage:md.append('| '+' | '.join(str(c[k]) if k in ['world','stage','mb'] else f'{c[k]:.3f}' for k in c)+' |')
 md+=['\n后三项在时间集合上闭合B包络，其他设备独占覆盖不等于所有其他算子的累计工作量。无归属设备覆盖不等于GPU空闲。各模块GPU并集可能重叠，不可直接相加。',
 '\nCP目前仍在Attention＋CP内，EP可能有不可见传输；归一化、激活、残差、拷贝及投影等尚未完整分类。此表不将未知项分配给Combine，也不证明对6.16%外推误差的因果贡献。']
 (a.out/'REPORT.md').write_text('\n'.join(md)+'\n')
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 (a.out/'manifest.json').write_text(json.dumps(dict(inputs={str(p):sha(p) for p in inputs},code={str(Path(__file__).resolve()):sha(Path(__file__)),str(Path(__file__).with_name('audit_b_operator_position_r1.py')):sha(Path(__file__).with_name('audit_b_operator_position_r1.py'))},outputs={str(p.resolve()):sha(p) for p in a.out.iterdir()}),indent=2)+'\n')
 print('PASS 70 module comparisons, 14 B coverage partitions; no raw trace rescan')
if __name__=='__main__':main()
