"""Explicit same-stream interval-union accounting for overlapping trace spans.

Raw events remain unchanged. Both raw span and exposed cost are retained;
this is not permission to clip arbitrary DAG dependency conflicts.
"""
import sys
import calibrate256_combined_r3 as combined
from calibrate256_stress_r2 import adapt as base_adapt
def adapt(module,text):
    text=base_adapt(module,text)
    if module=='build_b_cost_blocks_r1':
        text=text.replace("assert lo>=cursor-1e-5,(r,stream,lo,cursor)","assert hi>=cursor-1e-5,(r,stream,lo,hi,cursor)")
        before="g[k]=dict(deps=[gap],cost_ms=hi-lo,kind=kind"
        after="g[k]=dict(deps=[gap],cost_ms=hi-max(lo,cursor),observed_overlap_ms=max(0,cursor-lo),raw_span_ms=hi-lo,kind=kind"
        assert before in text;text=text.replace(before,after)
    if module=='build_b_recompute_cp_identity_r12':
        before="before=expected.get(prev,old.get(prev,{}).get('end_ms'));assert lo>=before-1e-5"
        after="before=expected.get(prev,old.get(prev,{}).get('end_ms'));raw_lo=lo;assert hi>=before-1e-5;lo=max(lo,before)"
        assert before in text;text=text.replace(before,after)
        before="blocks[k]=dict(start_ms=lo,end_ms=hi"
        after="g[k]['observed_overlap_ms']=max(0,before-raw_lo);g[k]['raw_start_ms']=raw_lo;blocks[k]=dict(start_ms=lo,end_ms=hi"
        assert before in text;text=text.replace(before,after)
    return text
if __name__=='__main__':
    combined.adapt=adapt
    combined.execute(combined.engine,"out=BASE/'calibrate256-fine-r1'","out=BASE/'calibrate256-overlap-r3'",[('text=code.read_text()','text=adapt(module,code.read_text())')])
    combined.execute(combined.normalizer,"root=BASE/'calibrate256-zero-r2'","root=BASE/'calibrate256-overlap-r3'")
    sys.argv+=['--model-dir',str(combined.engine.BASE/'iter60-shared-dag-r2'),'--tag','shared-binding-r3']
    combined.execute(combined.auditor,"root=BASE/'calibrate256-fine-r1'","root=BASE/'calibrate256-overlap-r3'",[("('B','b-recompute-r12')","('B','b-recompute-r12-zero-normalized')")])
