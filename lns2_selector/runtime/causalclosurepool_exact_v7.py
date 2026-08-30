from __future__ import annotations

import types
from bisect import bisect_left, bisect_right
from dataclasses import replace
from typing import Any, Iterable, Mapping

import lns2_selector.runtime.causalclosurepool as _reference
from experiments.state_analysis import StateAnalysis


CAUSALCLOSUREPOOL_EXACT_V7_ID = "stride-causalclosurepool-exact-v7"


def _clone_function_with_globals(
    function: Any, replacements: Mapping[str, Any]
) -> Any:
    namespace = dict(function.__globals__)
    namespace.update(replacements)
    cloned = types.FunctionType(
        function.__code__,
        namespace,
        name=function.__name__,
        argdefs=function.__defaults__,
        closure=function.__closure__,
    )
    cloned.__kwdefaults__ = function.__kwdefaults__
    cloned.__annotations__ = dict(function.__annotations__)
    return cloned


def generate_causalclosure_candidates_exact_v7(
    state: dict[str, Any],
    analysis: StateAnalysis,
    *,
    v2_anchors: Iterable[dict[str, Any]],
    maximum_candidates: int = 12,
    maximum_neighborhood_size: int = 64,
    temporal_window: int = 2,
    maximum_jaccard_similarity: float = 0.9,
    diagnostics: dict[str, int] | None = None,
) -> _reference.CausalClosurePoolResult:
    """Run the frozen CausalClosurePool with exact event-scan acceleration.

    All caches and replacement callables are local to this invocation.  The
    reference generator, its module globals, candidate semantics, ordering,
    and downstream temporal fixed point remain unchanged.
    """

    if diagnostics is not None:
        if diagnostics:
            raise ValueError("CausalClosure exact-v7 diagnostics must be empty")
        diagnostics.update(
            {
                "event_time_index_build_count": 0,
                "event_time_index_entry_count": 0,
                "event_slice_call_count": 0,
                "event_slice_candidate_count": 0,
                "event_slice_returned_count": 0,
                "direct_completion_count": 0,
                "oversized_closure_call_count": 0,
                "closure_call_count": 0,
            }
        )

    event_times_by_context: dict[int, dict[int, tuple[int, ...]]] = {}
    source_context_ids: set[int] = set()

    def indexed_with_events(
        context: _reference._CausalContext,
        call_analysis: StateAnalysis,
    ) -> _reference._CausalContext:
        if call_analysis is not analysis:
            raise RuntimeError("CausalClosure exact-v7 analysis identity changed")
        indexed = _reference._with_events(context, call_analysis)
        times = {
            int(agent): tuple(int(event.time) for event in events)
            for agent, events in indexed.events_by_agent.items()
        }
        event_times_by_context[id(indexed)] = times
        source_context_ids.add(id(indexed))
        if diagnostics is not None:
            diagnostics["event_time_index_build_count"] += 1
            diagnostics["event_time_index_entry_count"] += sum(
                len(values) for values in times.values()
            )
        return indexed

    def sliced_events(
        context: _reference._CausalContext,
        agent: int,
        causal_times: frozenset[int],
    ) -> tuple[Any, ...]:
        if id(context) not in source_context_ids:
            raise RuntimeError("CausalClosure exact-v7 context identity changed")
        events = context.events_by_agent.get(int(agent), ())
        if not events or not causal_times:
            if diagnostics is not None:
                diagnostics["event_slice_call_count"] += 1
                diagnostics["event_slice_candidate_count"] += len(events)
            return ()
        times = event_times_by_context[id(context)].get(int(agent), ())
        if len(times) != len(events):
            raise RuntimeError("CausalClosure exact-v7 event index changed")
        start_time = min(causal_times)
        stop_time = max(causal_times)
        start = bisect_left(times, start_time)
        stop = bisect_right(times, stop_time)
        selected = events[start:stop]
        if len(causal_times) != stop_time - start_time + 1:
            selected = tuple(
                event for event in selected if int(event.time) in causal_times
            )
        if diagnostics is not None:
            diagnostics["event_slice_call_count"] += 1
            diagnostics["event_slice_candidate_count"] += len(events)
            diagnostics["event_slice_returned_count"] += len(selected)
        return tuple(selected)

    def localized_with_sliced_events(
        context: _reference._CausalContext,
        core: set[int],
        seed_time: int,
        event_window: int,
    ) -> tuple[
        set[int],
        frozenset[int],
        dict[int, _reference.CausalContactEvidence],
    ]:
        eligible_times = set(
            range(
                max(0, int(seed_time) - int(event_window)),
                min(context.horizon, int(seed_time) + int(event_window) + 1),
            )
        )
        causal_times = frozenset(eligible_times)
        selected = set(core)
        stack = list(core)
        evidence: dict[int, _reference.CausalContactEvidence] = {}
        while stack:
            agent = stack.pop()
            for event in sliced_events(context, agent, causal_times):
                other = (
                    int(event.right)
                    if int(event.left) == agent
                    else int(event.left)
                )
                if other in selected:
                    continue
                selected.add(other)
                stack.append(other)
                evidence[other] = _reference._merge_evidence(
                    evidence.get(other), conflict_events=1
                )

        return selected, causal_times, evidence

    # The frozen generator invokes all five families consecutively for one
    # localized tuple.  A single strong-reference cache is therefore enough;
    # it prevents object-id reuse and bounds retained evidence to one core.
    last_raw_localized: tuple[Any, ...] | None = None
    last_completed_direct: tuple[
        set[int],
        frozenset[int],
        dict[int, _reference.CausalContactEvidence],
    ] | None = None

    def closure_with_empty_event_proxy(
        context: _reference._CausalContext,
        call_analysis: StateAnalysis,
        *,
        core_id: str,
        core_agents: tuple[int, ...],
        seed_time: int,
        family: str,
        temporal_radius: int | None,
        bottleneck_only: bool,
        event_window: int,
        maximum_neighborhood_size: int,
        localized: tuple[
            set[int],
            frozenset[int],
            dict[int, _reference.CausalContactEvidence],
        ]
        | None = None,
        temporal_neighbor_cache: dict[Any, Any] | None = None,
    ) -> tuple[Any, dict[str, Any] | None]:
        nonlocal last_raw_localized, last_completed_direct
        if call_analysis is not analysis:
            raise RuntimeError("CausalClosure exact-v7 closure analysis changed")
        if localized is None:
            localized = localized_with_sliced_events(
                context, set(core_agents), seed_time, event_window
            )
        if len(localized[0]) > maximum_neighborhood_size:
            # The frozen closure rejects here before scanning direct evidence.
            # Preserve that control-flow and cost boundary exactly.
            completed = localized
            if diagnostics is not None:
                diagnostics["oversized_closure_call_count"] += 1
        else:
            completed = (
                last_completed_direct
                if last_raw_localized is localized
                else None
            )
            if completed is None:
                selected = set(localized[0])
                causal_times = localized[1]
                evidence = dict(localized[2])
                core = set(core_agents)
                # The frozen closure adds every direct localized conflict to
                # the one discovery event already stored by localization.
                # Complete that same evidence once for this feasible core;
                # each family still receives the frozen closure's own mutable
                # list copy below.
                for agent in selected - core:
                    direct = sum(
                        (
                            (event.left in selected and event.right == agent)
                            or (event.right in selected and event.left == agent)
                        )
                        for event in sliced_events(
                            context, agent, causal_times
                        )
                    )
                    evidence[agent] = _reference._merge_evidence(
                        evidence.get(agent), conflict_events=int(direct)
                    )
                completed = (selected, causal_times, evidence)
                last_raw_localized = localized
                last_completed_direct = completed
                if diagnostics is not None:
                    diagnostics["direct_completion_count"] += 1
        # The reference closure only reads events_by_agent for the direct
        # evidence loop that was completed above.  An otherwise identical
        # proxy makes that loop add zero while retaining every frozen temporal
        # expansion and materialization operation.
        proxy = replace(context, events_by_agent={})
        if diagnostics is not None:
            diagnostics["closure_call_count"] += 1
        return _reference._closure_draft(
            proxy,
            call_analysis,
            core_id=core_id,
            core_agents=core_agents,
            seed_time=seed_time,
            family=family,
            temporal_radius=temporal_radius,
            bottleneck_only=bottleneck_only,
            event_window=event_window,
            maximum_neighborhood_size=maximum_neighborhood_size,
            localized=completed,
            temporal_neighbor_cache=temporal_neighbor_cache,
        )

    optimized = _clone_function_with_globals(
        _reference.generate_causalclosure_candidates,
        {
            "_with_events": indexed_with_events,
            "_localized_conflict_closure": localized_with_sliced_events,
            "_closure_draft": closure_with_empty_event_proxy,
        },
    )
    result = optimized(
        state,
        analysis,
        v2_anchors=v2_anchors,
        maximum_candidates=maximum_candidates,
        maximum_neighborhood_size=maximum_neighborhood_size,
        temporal_window=temporal_window,
        maximum_jaccard_similarity=maximum_jaccard_similarity,
    )
    if diagnostics is not None:
        diagnostics["event_slice_candidate_entries_skipped"] = (
            diagnostics.pop("event_slice_candidate_count")
            - diagnostics["event_slice_returned_count"]
        )
    return result


__all__ = [
    "CAUSALCLOSUREPOOL_EXACT_V7_ID",
    "generate_causalclosure_candidates_exact_v7",
]
