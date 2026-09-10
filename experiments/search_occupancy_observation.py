"""Map offered soft intervals to actual occupants outside native search."""
from collections import defaultdict


def at(path, tick):
    return path[min(tick, len(path)-1)]


class OccupancyIndex:
    def __init__(self, paths):
        self.vertices=defaultdict(set)
        self.edges=defaultdict(set)
        self.goals=defaultdict(list)
        for aid,path in paths.items():
            if not path: raise ValueError('empty occupancy path')
            for t,cell in enumerate(path):
                self.vertices[cell,t].add(aid)
                if t and path[t-1]!=cell:
                    self.edges[path[t-1],cell,t].add(aid)
            self.goals[path[-1]].append((len(path)-1,aid))

    def resolve(self,event):
        if len(event)!=10 or event[5]<0 or event[0] not in (0,1,2):
            raise ValueError('invalid offered interval')
        _,src,dst,_,_,tick,_,_,vertex,edge=event
        result={}
        if vertex:
            result['vertex']=self.vertices.get((dst,tick),set()) | {
                aid for arrival,aid in self.goals.get(dst,()) if tick>=arrival}
        if edge:
            result['edge']=set() if src==dst else self.edges.get((dst,src,tick),set())
        return result


def summarize_contacts(planner, fixed_paths, returned_path, events, external_ids):
    index=OccupancyIndex(fixed_paths)
    contacts=defaultdict(set); flagged=0; unmatched=[]
    for row in events:
        _,src,dst,_,_,tick,_,_,_,_=row
        for kind,owners in index.resolve(row).items():
            flagged+=1
            if not owners: unmatched.append(dict(event=row,kind=kind))
            on_returned=(at(returned_path,tick)==dst and
                         (kind=='vertex' or (tick>0 and at(returned_path,tick-1)==src)))
            if not on_returned:
                for owner in owners & external_ids:
                    contacts[owner].add((planner,kind,src,dst,tick))
    return dict(flagged=flagged,unmatched=unmatched,
                contacts={i:sorted(v) for i,v in contacts.items()})


def rank_contacts(summaries):
    by_owner=defaultdict(set)
    for summary in summaries:
        for owner,contacts in summary['contacts'].items():
            by_owner[int(owner)].update(tuple(x) for x in contacts)
    rows=[dict(agent=i,planner_count=len({x[0] for x in contacts}),
               contact_count=len(contacts)) for i,contacts in by_owner.items()]
    return sorted(rows,key=lambda r:(-r['planner_count'],-r['contact_count'],r['agent']))


class ObservingProbe:
    def __init__(self,module,probe,paths,external_ids,enabled,limit=65536):
        self.module=module; self.probe=probe; self.paths=paths; self.external=set(external_ids)
        self.enabled=enabled; self.limit=limit; self.captures=[]; self.summaries=[]

    def seed_rng(self,seed):
        self.probe.seed_rng(seed)

    def plan(self,agent,fixed,overrides,hard,*args,**kwargs):
        if not self.enabled: return self.probe.plan(agent,fixed,overrides,hard,*args,**kwargs)
        self.module.begin_observation(self.limit)
        try:
            result=self.probe.plan(agent,fixed,overrides,hard,*args,**kwargs)
        finally:
            capture=self.module.end_observation()
        self.captures.append(dict(agent=agent,fixed=fixed,capture=capture))
        if result['status']=='path':
            visible={i:overrides.get(i,self.paths[i]) for i in fixed}
            self.summaries.append(summarize_contacts(agent,visible,result['path'],capture['events'],self.external))
        return result
