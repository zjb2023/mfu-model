"""Strict CP association candidate; otherwise identical detailed builders."""
import inspect
from pathlib import Path
import calibrate256_fine_r1 as engine
from match_cp_marker_identity_r1 import match_marker
old=engine.split
text=inspect.getsource(old.extract)
before="pg=str(args['Process Group Name']);j=cpcount[pg];cpcount[pg]+=1\n            candidates=[m for m in markers if m['kind']=='CP' and m['members']==groups[pg]['ranks']];m=candidates[j]"
after="mi,marker,association=match_marker(es,r,e,f)\n            m=next(m for m in markers if m['cpu_event']==mi)\n            m['identity_association']=association"
assert text.count(before)==1;text=text.replace(before,after)
ns=dict(old.__dict__);ns.update(match_marker=match_marker,json=engine.FAST);exec(compile(text,__file__,'exec'),ns);old.extract=ns['extract']
code=Path(engine.__file__).read_text().replace("out=BASE/'calibrate256-fine-r1'","out=BASE/'calibrate256-strict-r2'")
state=dict(engine.__dict__);state['__name__']='strict_target_adapter';exec(compile(code,engine.__file__,'exec'),state)
if __name__=='__main__':state['main']()
