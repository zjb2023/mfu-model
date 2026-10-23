"""Combine strictly evidenced adapters in a new namespace; no r1 overwrite."""
import sys
from pathlib import Path
import calibrate256_fine_r1 as engine
# Import installs strict marker association in this process only.
import calibrate256_strict_r2
from calibrate256_stress_r2 import adapt
import normalize_zero_recompute_r2 as normalizer
import audit256_fine_binding_r1 as auditor
def execute(module,oldroot,newroot,changes=()):
    code=Path(module.__file__).read_text();assert oldroot in code;code=code.replace(oldroot,newroot)
    for before,after in changes:
        assert before in code;code=code.replace(before,after)
    ns=dict(module.__dict__);ns.update(__name__='combined_candidate',adapt=adapt)
    exec(compile(code,module.__file__,'exec'),ns);ns['main']()
if __name__=='__main__':
    execute(engine,"out=BASE/'calibrate256-fine-r1'","out=BASE/'calibrate256-combined-r3'",[('text=code.read_text()','text=adapt(module,code.read_text())')])
    execute(normalizer,"root=BASE/'calibrate256-zero-r2'","root=BASE/'calibrate256-combined-r3'")
    sys.argv+=['--model-dir',str(engine.BASE/'iter60-shared-dag-r2'),'--tag','shared-binding-r3']
    execute(auditor,"root=BASE/'calibrate256-fine-r1'","root=BASE/'calibrate256-combined-r3'",[("('B','b-recompute-r12')","('B','b-recompute-r12-zero-normalized')")])
