"""Apply the same normalization/audit to completed isolated adapter variants."""
import argparse,sys
import calibrate256_combined_r3 as c
from calibrate256_fine_r1 import BASE,load
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=['strict','stress'],required=True);ap.add_argument('--rank-base',type=int,required=True);ap.add_argument('--mb',type=int,required=True);a=ap.parse_args()
    rootname=f'calibrate256-{a.variant}-r2';root=BASE/rootname/f'r{a.rank_base}-mb{a.mb}';assert (root/'scope.json').exists()
    sys.argv=[__file__,'--rank-base',str(a.rank_base),'--mb',str(a.mb)]
    if not (root/'b-recompute-r12-zero-normalized/report.json').exists():c.execute(c.normalizer,"root=BASE/'calibrate256-zero-r2'",f"root=BASE/'{rootname}'")
    sys.argv+=['--model-dir',str(BASE/'iter60-shared-dag-r2'),'--tag','shared-binding-r3']
    if not (root/'shared-binding-r3/report.json').exists():c.execute(c.auditor,"root=BASE/'calibrate256-fine-r1'",f"root=BASE/'{rootname}'",[("('B','b-recompute-r12')","('B','b-recompute-r12-zero-normalized')")])
    print(load(root/'shared-binding-r3/report.json')['phases'].keys())
if __name__=='__main__':main()
