"""Identity-explicit A/A2 confirmation; no changes to the frozen timer or policies."""
from experiments import sa_expanded_timing as old
from experiments.sa_raw_selection_fast import _bind

ARMS = old.ARMS
LABELS = dict(old.LABELS, parent='Frozen A + SA (iteration 2)',
              completion='Frozen A2 + SA (iteration 3; exploratory)')
TIMING_MODE = old.TIMING_MODE
qualification_worker = old.qualification_worker
preflight_worker = old.preflight_worker
timed_worker = old.timed_worker
audit_worker = old.audit_worker
summarize = _bind(old.summarize, LABELS=LABELS)
