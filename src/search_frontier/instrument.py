"""Generate an isolated compilation unit with three exact, reversible hooks."""
import hashlib
from pathlib import Path
import sys

SOURCE_SHA = 'ff38ffb74a380be5d6e3833b15540c5cb3baf7a70ed0866d4abdd1136fe41a85'
HOOKS = (
    ('void SIPP::updatePath(const LLNode* goal, vector<PathEntry> &path)\n{',
     '\n    frontier_observer::finish(goal);'),
    ('        num_expanded++;\n        assert(curr->location >= 0);',
     '\n        frontier_observer::record(2, curr);'),
    ('    node->focal_handle = focal_list.push(node); // we only use focal list; no open list is used',
     '\n    frontier_observer::record(1, node);'),
)


def transform(source):
    result = source
    for index, (anchor, addition) in enumerate(HOOKS):
        # The pop sequence also occurs in findNoCollisionPath; instrument findPath only.
        if result.count(anchor) != (2 if index == 1 else 1):
            raise ValueError('ambiguous or missing observation hook')
        result = result.replace(anchor, anchor + addition, 1)
    restored = result
    for anchor, addition in reversed(HOOKS):
        restored = restored.replace(anchor + addition, anchor)
    if restored != source:
        raise ValueError('non-observation source changes')
    return '#include "observer.h"\n' + result


if __name__ == '__main__':
    source, output = map(Path, sys.argv[1:])
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError('frozen SIPP source SHA changed')
    output.write_text(transform(source.read_text()), encoding='utf-8')
