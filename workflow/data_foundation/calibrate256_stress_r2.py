"""Scale only the CP perturbation TEST to exceed measured readiness slack.

The model costs and dependency construction are unchanged. This can distinguish
a nonbinding 2ms perturbation from a genuine missing coupling edge.
"""
from pathlib import Path
import calibrate256_fine_r1 as engine
from zero_work_evidence_r2 import adapt as zero_adapt
def adapt(module,text):
    text=zero_adapt(module,text)
    if module=='build_b_cp_pair_r4':
        before="c=copy.deepcopy(g);c[flash[slow]]['cost_ms']+=2;t=run(c)"
        after="increment=max(2,max(v['end_ms'] for v in baseline.values())-baseline[flash[slow]]['end_ms']+2);c=copy.deepcopy(g);c[flash[slow]]['cost_ms']+=increment;t=run(c)"
        assert text.count(before)==1;text=text.replace(before,after).replace('attention_increment_ms=2','attention_increment_ms=increment')
    return text
code=Path(engine.__file__).read_text().replace("out=BASE/'calibrate256-fine-r1'","out=BASE/'calibrate256-stress-r2'")
code=code.replace('text=code.read_text()','text=adapt(module,code.read_text())')
ns=dict(engine.__dict__);ns.update(__name__='stress_adapter',adapt=adapt);exec(compile(code,engine.__file__,'exec'),ns)
if __name__=='__main__':ns['main']()
