"""Small, offline path-time representation probe; never a runtime controller."""
from collections import Counter
import math

import numpy as np

from experiments.sa_paired_completion import require
from experiments.sa_spatiotemporal_input import decode_paths

BASE_CHANNELS = 27
CONDITION_CHANNELS = 32
PROFILES = ("flat", "time_bag", "path_time")


def _occupancy(paths, vertices, parked=None):
    counts = np.zeros((paths.shape[1], vertices), dtype=np.int32)
    times = np.broadcast_to(np.arange(paths.shape[1]), paths.shape)
    take = np.ones(paths.shape, dtype=bool) if parked is None else parked
    np.add.at(counts, (times[take], paths[take]), 1)
    return counts


def _moves(paths, vertices):
    keys = []
    for t in range(1, paths.shape[1]):
        keys.extend((t, int(a)*vertices+int(b)) for a,b in zip(paths[:,t-1],paths[:,t]) if a != b)
    return Counter(keys)


def encode_model_input(encoded, memberships):
    """Exact-time, one-cell path exposure; not an unbounded spatial encoder.

    All outsider paths form the occupancy fields, but the network sees those
    fields only at selected paths and their one-cell spatial neighbors.
    """
    agents = decode_paths(encoded)
    ids = [a["id"] for a in agents]
    require(memberships and all(s and len(s)==len(set(s)) and set(s)<=set(ids) for s in memberships),
            "invalid candidate grid")
    require(len({tuple(sorted(s)) for s in memberships})==len(memberships), "duplicate candidate set")
    rows, cols = encoded["rows"], encoded["cols"]
    vertices, length = rows*cols, encoded["horizon"]+1
    paths = np.array([a["path"]+[a["goal"]]*(length+1-len(a["path"])) for a in agents], dtype=np.int64)
    arrival = np.array([a["occupancy"][-1][1] for a in encoded["agents"]])
    parked = np.arange(length+1)[None,:] >= arrival[:,None]
    counts, parking = _occupancy(paths,vertices), _occupancy(paths,vertices,parked)
    moves = _moves(paths,vertices)
    union = sorted(set().union(*(set(s) for s in memberships)))
    source_rows = np.array([ids.index(i) for i in union])
    points = paths[source_rows,:length]
    yy,xx = points//cols,points%cols
    query_y = yy[...,None] + np.array([0,-1,1,0,0])
    query_x = xx[...,None] + np.array([0,0,0,-1,1])
    valid = (query_y>=0)&(query_y<rows)&(query_x>=0)&(query_x<cols)
    cells = np.clip(query_y,0,rows-1)*cols+np.clip(query_x,0,cols-1)
    free = valid & (np.array(encoded["obstacles"])[cells]==0)
    times = np.arange(length)[None,:,None]

    def query(field, offset=0):
        return field[times+offset,cells]*free

    own = (cells==points[...,None])*free
    own_next = (cells==paths[source_rows,1:length+1,None])*free
    global_here = query(counts)-own
    global_next = query(counts,1)-own_next
    previous = np.concatenate([points[:,:1],points[:,:-1]],axis=1)
    dy,dx = yy-previous//cols,xx-previous%cols
    reverse = np.array([[moves.get((t,int(points[a,t])*vertices+int(previous[a,t])),0)
                         if points[a,t]!=previous[a,t] else 0 for t in range(length)] for a in range(len(union))])
    shape = points.shape
    coordinate = lambda v:np.broadcast_to(v,shape)
    costs = np.array([len(agents[i]["path"])-1 for i in source_rows])
    starts,goals = paths[source_rows,0],paths[source_rows,-1]
    basic = [yy/max(1,rows-1),xx/max(1,cols-1),dy,dx,(points==previous),parked[source_rows,:length],
             coordinate(np.arange(length)[None,:]/256),coordinate(costs[:,None]/256),
             coordinate((goals//cols)[:,None]/max(1,rows-1)),coordinate((goals%cols)[:,None]/max(1,cols-1)),
             coordinate((starts//cols)[:,None]/max(1,rows-1)),coordinate((starts%cols)[:,None]/max(1,cols-1))]
    base = np.concatenate([np.stack(basic,axis=-1),1-free[:,:,1:],np.log1p(global_here),
                           np.log1p(global_next),np.log1p(reverse)[...,None]],axis=-1).astype(np.float32)
    conditions, indices = [], []
    for membership in memberships:
        selected = [ids.index(a) for a in sorted(membership)]
        local = [union.index(a) for a in sorted(membership)]
        selected_counts = _occupancy(paths[selected],vertices)
        selected_parking = _occupancy(paths[selected],vertices,parked[selected])
        selected_moves = _moves(paths[selected],vertices)
        inside = query(selected_counts)-own
        inside_next = query(selected_counts,1)-own_next
        # Only selected agents' rows are retained, so subtracting own occupancy
        # is correct even though temporary arrays contain unselected union rows.
        own_park = own * parked[source_rows,:length,None]
        inside_park = query(selected_parking)-own_park
        sr = np.array([[selected_moves.get((t,int(points[a,t])*vertices+int(previous[a,t])),0)
                        if points[a,t]!=previous[a,t] else 0 for t in range(length)] for a in local])
        channels = np.concatenate([inside[local],query(counts-selected_counts)[local],
            inside_next[local],query(counts-selected_counts,1)[local],inside_park[local],
            query(parking-selected_parking)[local],sr[...,None],(reverse[local]-sr)[...,None]],axis=-1)
        require(np.all(channels>=0),"negative conditioned occupancy")
        conditions.append(np.log1p(channels).astype(np.float32))
        indices.append(np.array(local,dtype=np.int64))
    require(base.shape[-1]==BASE_CHANNELS and all(c.shape[-1]==CONDITION_CHANNELS for c in conditions),
            "token schema mismatch")
    require(np.isfinite(base).all() and all(np.isfinite(c).all() for c in conditions),"nonfinite input")
    return dict(base=base,condition=conditions,indices=indices,length=length)


def fit_scaler(states, held, names):
    train = [s for s in states if s["map_id"]!=held]
    require(train and any(s["map_id"]==held for s in states),"invalid held map")
    x = np.array([[c["features"][n] for n in names] for s in train for c in s["candidates"]],dtype=np.float64)
    mean,std = x.mean(axis=0),x.std(axis=0)
    std[std<1e-8] = 1
    return mean.astype(np.float32),std.astype(np.float32)


def select(scores, candidates, anchor):
    require(len(scores)==len(candidates) and all(math.isfinite(float(v)) for v in scores),"invalid scores")
    return min(range(len(candidates)),key=lambda i:(-float(scores[i]),candidates[i]!=anchor,candidates[i]))


def make_model(profile, flat_dim, hidden=16):
    import torch
    from torch import nn
    require(profile in PROFILES,"unknown neural profile")

    class PathTimeProbe(nn.Module):
        def __init__(self):
            super().__init__()
            if profile!="flat":
                self.path1 = nn.Conv1d(BASE_CHANNELS,hidden,3,padding=1)
                self.path2 = nn.Conv1d(hidden,hidden,3,padding=4,dilation=4)
                self.context = nn.Conv1d(hidden+CONDITION_CHANNELS,hidden,3,padding=1)
            self.head = nn.Sequential(nn.Linear(flat_dim+(2*hidden if profile!="flat" else 0),32),
                                      nn.GELU(),nn.Linear(32,1))

        def forward(self, batch, flat):
            if profile=="flat":
                return self.head(flat).squeeze(-1).sigmoid()
            base,condition = batch["base"],batch["condition"]
            b,n,t,f = base.shape
            _,c,k,_,_ = condition.shape
            tm = batch["time_mask"]
            path_mask = tm[:,None,:,None]
            cm = tm[:,None,None,:,None]
            if profile=="time_bag":
                base = (base*path_mask).sum(2,keepdim=True)/path_mask.sum(2,keepdim=True).clamp_min(1)
                base = base.expand(-1,-1,t,-1)*path_mask
                condition = (condition*cm).sum(3,keepdim=True)/cm.sum(3,keepdim=True).clamp_min(1)
                condition = condition.expand(-1,-1,-1,t,-1)*cm
            x = base.permute(0,1,3,2).reshape(b*n,f,t)
            pm = tm[:,None,None,:].expand(b,n,1,t).reshape(b*n,1,t)
            x = torch.nn.functional.gelu(self.path1(x))*pm
            x = torch.nn.functional.gelu(self.path2(x))*pm
            x = x.reshape(b,n,hidden,t).permute(0,1,3,2)
            picked = x[torch.arange(b,device=x.device)[:,None,None],batch["indices"]]
            joined = torch.cat([picked,condition],dim=-1).permute(0,1,2,4,3).reshape(b*c*k,hidden+CONDITION_CHANNELS,t)
            z = torch.nn.functional.gelu(self.context(joined)).reshape(b,c,k,hidden,t).permute(0,1,2,4,3)
            mask = batch["agent_mask"][:,:,:,None,None]*cm
            avg = (z*mask).sum((2,3))/mask.sum((2,3)).clamp_min(1)
            maximum = z.masked_fill(~mask.bool(),float("-inf")).amax((2,3))
            return self.head(torch.cat([flat,avg,maximum],dim=-1)).squeeze(-1).sigmoid()

    return PathTimeProbe()
