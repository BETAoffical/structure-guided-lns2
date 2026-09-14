"""Evidence-only source snapshots and representation-independent edge checks."""
from pathlib import Path
import shutil

from experiments._common import sha256_file,read_json
from lns2_selector.runtime.fingerprints import semantic_fingerprint


def read_bound_plan(root,directory,evidence_only=False):
    directory=Path(directory)
    plan=read_json(directory/"plan.json")
    if plan["binding"]!=semantic_fingerprint({k:v for k,v in plan.items() if k!="binding"}):
        raise ValueError("plan changed")
    verify_inputs(root,plan["inputs"],directory/"registered_sources" if evidence_only else None)
    return plan


def pair_absent(edges, pair):
    target=tuple(sorted(pair))
    return target not in {tuple(sorted(e)) for e in edges}


def snapshot_sources(root, inputs, directory):
    root,directory=Path(root),Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    for name,digest in inputs.items():
        path=root/name
        if path.suffix not in (".py",".md"):
            continue
        if sha256_file(path)!=digest:
            raise ValueError("cannot snapshot changed source: "+name)
        destination=directory/(digest+path.suffix)
        if destination.exists():
            if sha256_file(destination)!=digest:
                raise ValueError("snapshot changed")
        else:
            shutil.copyfile(path,destination)


def verify_inputs(root, inputs, snapshots=None):
    """Snapshots may be used to read old evidence, never to authorize new execution."""
    used=[]
    root=Path(root)
    for name,digest in inputs.items():
        path=root/name
        if path.exists() and sha256_file(path)==digest:
            continue
        snapshot=None if snapshots is None else Path(snapshots)/(digest+path.suffix)
        if (path.suffix not in (".py",".md") or snapshot is None or not snapshot.is_file()
                or sha256_file(snapshot)!=digest):
            raise ValueError("registered input changed: "+name)
        used.append(name)
    return used
