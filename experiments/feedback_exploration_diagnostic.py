"""Experimental feedback selection only; never imported by production controllers."""

from collections import defaultdict
import random


class FeedbackSelection:
    ARMS = ("frozen", "uniform", "feedback")

    def __init__(self, arm):
        if arm not in self.ARMS:
            raise ValueError("unknown diagnostic arm")
        self.arm = arm
        self.observations = defaultdict(lambda: [0, 0])

    def select(self, keys, candidate_ids, base_index, seed):
        if not keys or len(keys) != len(candidate_ids) or len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("invalid candidates")
        if not 0 <= base_index < len(keys):
            raise ValueError("invalid base index")
        prior = [list(self.observations.get(key, (0, 0))) for key in keys]
        active = self.arm != "frozen" and prior[base_index][1] > 0
        samples = None
        choice = base_index
        if active:
            rng = random.Random(seed)
            ordered = sorted(range(len(keys)), key=lambda i: candidate_ids[i])
            if self.arm == "uniform":
                choice = ordered[rng.randrange(len(ordered))]
            else:
                samples = [0.0] * len(keys)
                for i in ordered:
                    success, nondrop = prior[i]
                    samples[i] = rng.betavariate(1 + success, 1 + nondrop)
                choice = min(ordered, key=lambda i: (-samples[i], candidate_ids[i]))
        return choice, {"active": active, "prior": prior, "samples": samples}

    def observe(self, key, before_conflicts, after_conflicts, censored):
        if censored:
            return
        if after_conflicts > before_conflicts:
            raise ValueError("unexpected accepted conflict increase")
        self.observations[key][0 if after_conflicts < before_conflicts else 1] += 1


def padded_auc(conflicts, horizon):
    if not conflicts or len(conflicts) > horizon + 1:
        raise ValueError("invalid conflict series")
    extended = list(conflicts) + [conflicts[-1]] * (horizon + 1 - len(conflicts))
    return sum((a + b) / 2 for a, b in zip(extended, extended[1:]))


def paired_summary(rows, arms=FeedbackSelection.ARMS):
    identities = [(r["arm"], r["case_id"], r["trial"]) for r in rows]
    if len(set(identities)) != len(rows) or any(r["arm"] not in arms for r in rows):
        raise ValueError("duplicate or unknown arm")
    tables = {arm: {(r["case_id"], r["trial"]): r for r in rows if r["arm"] == arm} for arm in arms}
    if not tables["frozen"] or any(set(v) != set(tables["frozen"]) for v in tables.values()):
        raise ValueError("unpaired stage")
    summaries = {}
    for arm, table in tables.items():
        pairs = [(tables["frozen"][key], table[key]) for key in sorted(table)]
        summaries[arm] = {
            "episodes": len(pairs), "feasible": sum(b["feasible"] for a, b in pairs),
            "paired_feasibility_losses": sum(a["feasible"] and not b["feasible"] for a, b in pairs),
            "paired_feasibility_gains": sum(not a["feasible"] and b["feasible"] for a, b in pairs),
            "auc_wins": sum(b["auc"] < a["auc"] for a, b in pairs),
            "auc_sum": sum(b["auc"] for a, b in pairs),
            "final_conflict_sum": sum(b["conflicts"][-1] for a, b in pairs),
            "censored": sum(b["censored"] for a, b in pairs),
            "feedback_active_steps": sum(e["feedback"]["active"] for a, b in pairs for e in b["transitions"]),
        }
    return summaries


def classify_rounds(discovery, validation):
    """Fixed development-only criteria, never a production promotion gate."""
    stages = (discovery, validation)
    if any(s[a]["censored"] for s in stages for a in FeedbackSelection.ARMS):
        return "inconclusive_resource"
    def recovery(arm):
        return (discovery[arm]["paired_feasibility_gains"] >= 2 and
                all(s[arm]["paired_feasibility_losses"] == 0 for s in stages))
    incremental = (recovery("feedback") and
                   all(s["feedback"]["feasible"] >= s["uniform"]["feasible"] for s in stages) and
                   sum(s["feedback"]["feasible"] for s in stages) > sum(s["uniform"]["feasible"] for s in stages))
    if incremental:
        return "feedback_incremental_development_signal"
    if recovery("uniform") or recovery("feedback"):
        return "exploration_signal_without_feedback_increment"
    return "no_consistent_recovery_signal"
