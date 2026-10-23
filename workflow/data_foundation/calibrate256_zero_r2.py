"""Experimental zero-work adapter, isolated from running/frozen r1 outputs."""
from pathlib import Path
import calibrate256_fine_r1 as engine
from zero_work_evidence_r2 import adapt
code=Path(engine.__file__).read_text()
assert "out=BASE/'calibrate256-fine-r1'" in code
code=code.replace("out=BASE/'calibrate256-fine-r1'","out=BASE/'calibrate256-zero-r2'")
code=code.replace("text=code.read_text()","text=adapt(module,code.read_text())")
code=code.replace("adapter=out/(name+'.adapter.py')","adapter=out/(name+'.adapter-r2c.py')")
code=code.replace(".open('w') as log", ".open('a') as log")
ns=dict(engine.__dict__);ns['__name__']='zero_calibration_adapter';ns['adapt']=adapt
exec(compile(code,engine.__file__,'exec'),ns)
if __name__=='__main__':ns['main']()
