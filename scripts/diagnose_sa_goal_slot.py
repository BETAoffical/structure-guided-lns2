"""Record independent goal-conditioned proofs for late CBS obstructions."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import diagnose_sa_pair_compatibility as source
from experiments.goal_slot_certificate import prove_goal_slot
from experiments.diagnostic_integrity import verify_inputs

io=source.io
OUT=ROOT/"build/sa-goal-obstruction-v1"


def main():
    io.require(not (OUT/"report.json").exists(),"preserve prior goal proof")
    plan=source.verify(evidence_only=True)
    old_path=ROOT/"build/sa-external-obstruction-v1/report.json"
    old=io.read(old_path)
    verify_inputs(ROOT,old["inputs"])
    proofs=[]
    for proof in old["proofs"]:
        if proof["status"]!="no_early_prefix_certificate": continue
        case=next(c for c in plan["cases"] if c["id"]==proof["job"]["case_id"])
        certificate=prove_goal_slot(case["state"],case["selected"],proof["job"]["pair"],proof["event"]["time"])
        proofs.append(dict(job=proof["job"],event=proof["event"],certificate=certificate,
            status="forced_goal_slot" if certificate["forced_collision"] else "not_proved",
            blocker_candidates=certificate["blockers"]))
    names=["scripts/diagnose_sa_goal_slot.py","experiments/goal_slot_certificate.py",old_path.relative_to(ROOT).as_posix()]
    report=dict(schema="lns2.sa_goal_obstruction.v1",proofs=proofs,no_ttf=True,
                inputs={n:io.sha256_file(ROOT/n) for n in names})
    io.write(OUT/"report.json",report)
    print(proofs)


if __name__=="__main__":
    main()
