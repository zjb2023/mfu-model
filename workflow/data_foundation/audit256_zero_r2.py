"""Run the same frozen-edge audit against the isolated zero-work candidate."""
from pathlib import Path
import audit256_fine_binding_r1 as engine
code=Path(engine.__file__).read_text()
assert "root=BASE/'calibrate256-fine-r1'" in code
code=code.replace("root=BASE/'calibrate256-fine-r1'","root=BASE/'calibrate256-zero-r2'")
code=code.replace("('B','b-recompute-r12')","('B','b-recompute-r12-zero-normalized')")
ns=dict(engine.__dict__);ns['__name__']='zero_audit_adapter'
exec(compile(code,engine.__file__,'exec'),ns)
if __name__=='__main__':ns['main']()
