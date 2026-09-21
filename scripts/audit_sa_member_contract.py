"""Audit deployment/label coverage for member-only correction; never solve or fit."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import contained_file, json_fingerprint, read_json, sha256_file, write_json
from experiments.sa_paired_completion import require
from scripts.prepare_sa_source_matched import grouped, signature

EVIDENCE = "artifacts/sa-matched-aggregation-v1/evidence.json"
EVIDENCE_SHA = "cac7808403da44a65fe4f0c0b68671f4fddbd27e51bc3d49e3af6bb7e17492b7"
SOURCE = "build/sa-matched-aggregation-v1"
OUTPUT = "build/sa-member-decision-contract-v1/report.json"
OFFICIAL16 = {"target:16", "collision:16", "random:16"}


def load_pinned(root, name, expected, inputs):
    path = contained_file(root, name, field="member-contract input")
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    require(digest == expected, "input drift: " + name)
    require(name not in inputs or inputs[name] == digest, "conflicting input identity")
    inputs[name] = digest
    return json.loads(payload)


def scope_row(entry, root, groups):
    """Scope uses pre-action membership/provenance, never scores or outcomes."""
    sid, anchor = entry["state_id"], entry["anchor_id"]
    event = root["control_event"]
    pool = event["pool"]
    grouped(pool, 16)  # Reuse source audit's full-pool identity validation.
    candidates = {c["candidate_id"]: c for c in pool}
    require(anchor in candidates and anchor == root["old_selected_id"], "missing or different anchor")
    require(type(event["selected_index"]) is int and 0 <= event["selected_index"] < len(pool), "invalid selected index")
    require(pool[event["selected_index"]]["candidate_id"] == anchor and
            event["action"]["agents"] == candidates[anchor]["agents"], "anchor action mismatch")
    source = root["source"]
    require((source["id"], source["map_id"], source["decision"], event["decision"]) ==
            (sid, entry["map_id"], entry["decision"], entry["decision"]), "root identity mismatch")
    require(all(set(c["agents"]) <= set(entry["agent_ids"]) for c in pool), "unknown candidate agent")
    anchor_signature = signature(candidates[anchor])
    alternatives = sorted(cid for cid, c in candidates.items() if cid != anchor and signature(c) == anchor_signature)
    covered = set()
    indexed_pairs = []
    matched_anchor = False
    matched_same_signature = False
    for group in groups:
        require(group["state_id"] == sid and group["map_id"] == entry["map_id"], "group state mismatch")
        rows = group["candidates"]
        ids = [c["candidate_id"] for c in rows]
        require(len(ids) == len(set(ids)) and set(ids) <= candidates.keys(), "unknown or duplicate label candidate")
        require(len(group["trial_keys"]) == 8 and len(set(group["trial_keys"])) == 8, "trial stream coverage")
        for c in rows:
            cid = c["candidate_id"]
            require(sorted(c["agents"]) == sorted(candidates[cid]["agents"]), "label member mismatch")
            require(len(c["values"]) == 8 and all(type(v) is bool for v in c["values"]), "incomplete labels")
        if group["kind"] == "matched_pair":
            require(len(ids) == 2 and group["family"] in OFFICIAL16 and
                    all(signature(candidates[cid]) == (16, (group["family"],)) for cid in ids), "matched signature mismatch")
            matched_anchor = anchor in ids
            matched_same_signature = all(signature(candidates[cid]) == anchor_signature for cid in ids)
        else:
            require(group["kind"] == "old_grid" and len(ids) == 4, "unknown label group")
        # Trial index alone cannot connect labels from different collection groups.
        if anchor in ids:
            for cid in sorted(set(ids) & set(alternatives)):
                covered.add(cid)
                indexed_pairs.append(dict(group_id=group["group_id"], kind=group["kind"],
                                          anchor=anchor, alternate=cid,
                                          trial_stream=json_fingerprint(group["trial_keys"])))
    families = list(anchor_signature[1])
    official16 = anchor_signature[0] == 16 and len(families) == 1 and families[0] in OFFICIAL16
    return dict(state_id=sid, map_id=entry["map_id"], decision=entry["decision"],
                anchor_id=anchor, anchor_actual_size=anchor_signature[0], anchor_families=families,
                structural_anchor=any(f.startswith("structpool-") for f in families),
                same_signature_alternatives=alternatives,
                eligible=bool(alternatives), official16_eligible=bool(alternatives) and official16,
                matched_pair_contains_anchor=matched_anchor,
                matched_pair_has_anchor_signature=matched_same_signature,
                indexed_labeled_alternatives=sorted(covered),
                indexed_pairs=sorted(indexed_pairs, key=lambda p: (p["group_id"], p["alternate"])),
                unlabeled_alternatives=sorted(set(alternatives) - covered))


def summarize(rows):
    return dict(roots=len(rows), maps=len({r["map_id"] for r in rows}),
                eligible_roots=sum(r["eligible"] for r in rows),
                official16_eligible_roots=sum(r["official16_eligible"] for r in rows),
                no_alternative_roots=sum(not r["eligible"] for r in rows),
                structural_anchor_roots=sum(r["structural_anchor"] for r in rows),
                structural_eligible_roots=sum(r["structural_anchor"] and r["eligible"] for r in rows),
                matched_contains_anchor_roots=sum(r["matched_pair_contains_anchor"] for r in rows),
                indexed_label_covered_roots=sum(bool(r["indexed_labeled_alternatives"]) for r in rows),
                eligible_but_unlabeled_roots=sum(r["eligible"] and not r["indexed_labeled_alternatives"] for r in rows),
                anchor_signatures=dict(sorted(Counter(
                    str(r["anchor_actual_size"]) + "/" + "+".join(r["anchor_families"]) for r in rows).items())))


def audit(root=ROOT):
    inputs = {}
    evidence = load_pinned(root, EVIDENCE, EVIDENCE_SHA, inputs)
    files = {name: load_pinned(root, name, digest, inputs) for name, digest in evidence["files"].items()
             if name.endswith(".json")}
    plan, index = files[SOURCE + "/plan.json"], files[SOURCE + "/training_index.json"]
    require(plan["binding"] == evidence["binding"] == index["binding"] ==
            json_fingerprint({k: v for k, v in plan.items() if k != "binding"}), "source binding mismatch")
    require(index["integrity"] == json_fingerprint({k: v for k, v in index.items() if k != "integrity"}), "index integrity")
    entries = index["entries"]
    roots = plan["roots"]
    require(len(roots) == len(entries) == 47 and len({r["state_id"] for r in roots}) == 47 and
            set(entries) == {r["state_id"] for r in roots}, "source root coverage")
    require(len(index["groups"]) == 94 and len({g["group_id"] for g in index["groups"]}) == 94 and
            {g["state_id"] for g in index["groups"]} == set(entries), "label group coverage")
    rows = []
    for r in sorted(roots, key=lambda r: (r["map_id"], r["state_id"])):
        sid, name = r["state_id"], r["source_root"]
        require(name == entries[sid]["source_root"] and r["map_id"] == entries[sid]["map_id"] and
                r["anchor_id"] == entries[sid]["anchor_id"] and r["decision"] == entries[sid]["decision"], "entry drift")
        source_root = load_pinned(root, name, plan["inputs"][name], inputs)
        require(source_root["state_fingerprint"] == r["root_fingerprint"] == source_root["source"]["state_fingerprint"] and
                source_root["binding"] == r["source_binding"], "root fingerprint or binding drift")
        groups = [g for g in index["groups"] if g["state_id"] == sid]
        require(sorted(g["kind"] for g in groups) == ["matched_pair", "old_grid"], "duplicate or missing group kind")
        rows.append(scope_row(entries[sid], source_root, groups))
    require(len({r["map_id"] for r in rows}) == 8, "map coverage")
    # Extra anchor outcomes from the first 16 roots were not training groups.
    # Check their identities separately; never combine them with old-grid streams.
    diagnostic_path = "build/sa-source-matched-resource-recovery-v1/analysis_records.json"
    diagnostic = load_pinned(root, diagnostic_path, plan["inputs"][diagnostic_path], inputs)
    require(len(diagnostic) == len({d["state_id"] for d in diagnostic}) == 16, "diagnostic root coverage")
    by_id = {r["state_id"]: r for r in rows}
    extra = []
    for d in diagnostic:
        row = by_id[d["state_id"]]
        require(d["anchor_id"] == row["anchor_id"] and d["map_id"] == row["map_id"], "diagnostic anchor mismatch")
        group = next(g for g in index["groups"] if g["state_id"] == row["state_id"] and g["kind"] == "matched_pair")
        require(sorted(d["pair_ids"]) == sorted(c["candidate_id"] for c in group["candidates"]) and
                set(d["values"]) == set(d["pair_ids"]) | {row["anchor_id"]}, "diagnostic candidate mismatch")
        require(all(d["values"][c["candidate_id"]] == c["values"] for c in group["candidates"]) and
                all(len(v) == 8 and all(type(x) is bool for x in v) for v in d["values"].values()), "diagnostic labels mismatch")
        extra.extend([row["state_id"], cid] for cid in row["unlabeled_alternatives"] if cid in d["values"])
    for name, digest in inputs.items():
        require(sha256_file(contained_file(root, name, field="audit input")) == digest, "input changed during audit")
    return dict(schema="lns2.sa.member_decision_contract.v1", role="viewed_development_scope_not_performance",
                source_binding=plan["binding"], inputs=inputs, summary=summarize(rows), rows=rows,
                by_map={m: summarize([r for r in rows if r["map_id"] == m]) for m in sorted({r["map_id"] for r in rows})},
                diagnostic_anchor_records_checked=len(diagnostic), extra_diagnostic_pairs=sorted(extra),
                model_fits=0, solver_calls=0, new_labels=0, runtime_modified=False,
                evaluated_policy_performance=False, continuous_control_identified=False,
                training_ready=False, decision="member_only_successor_not_supported_by_current_contract")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Recompute and compare the existing coverage report.")
    args = parser.parse_args()
    report = audit()
    output = ROOT / OUTPUT
    require(not output.is_symlink() and not output.parent.is_symlink(), "unsafe output")
    require(output.resolve().is_relative_to((ROOT / "build").resolve()), "output outside build")
    if args.verify or output.exists():
        require(read_json(output) == report, "existing report differs; never overwrite")
    else:
        write_json(output, report)
    print(json.dumps(report["summary"], indent=2))
    print("diagnostic_extra_pairs", len(report["extra_diagnostic_pairs"]))
    print("report_sha256", sha256_file(output))


if __name__ == "__main__":
    main()
