import math
import torch

@torch.no_grad()
def build_event_only_relations_skip_missing(
    data,
    time_attr: str = 'time',          # integer trading-day index on event nodes
    use_time_decay: bool = True,
    half_life_same: float = 90.0,
    half_life_peer: float = 60.0,
    K_same: int | None = None,        # None = keep all prior events available
    K_peer: int | None = None,
    treat_sim_undirected: bool = False  # if True, also use reverse similar edges
):
    """
    Create event-only relations once on the full HeteroData:
      ('event','prev_same','event')  : previous events of the *same* firm  -> event
      ('event','prev_peer','event')  : previous events of *similar* firms  -> event

    Missing firm links:
      - If an event lacks an ('event','of','firm') edge, it receives no same/peer edges.
      - Source-side events come only from existing ('firm','rev_of','event') edges.

    Assumes:
      - data[('event','of','firm')].edge_index exists (possibly incomplete).
      - data[('firm','rev_of','event')].edge_index lists firm→its events you want to consider.
      - data[('firm','similar','firm')].edge_index lists similar firms (weights optional/ignored).
    """
    dev = data['event'].x.device
    Ne  = data['event'].num_nodes
    Nf  = data['firm'].num_nodes

    # --- event time (Long)
    day = getattr(data['event'], time_attr).long().to(dev)

    # --- event -> firm (may be missing for some events)
    e2f = data[('event','of','firm')].edge_index.to(dev)        # [2, E_e2f]
    firm_of_event = torch.full((Ne,), -1, dtype=torch.long, device=dev)
    firm_of_event[e2f[0]] = e2f[1]                               # events w/o firm stay -1

    # --- firm -> events that exist in the graph (use this to source "previous events")
    f2e = data[('firm','rev_of','event')].edge_index.to(dev)     # [2, E_f2e], src=firm, dst=event
    # bucket events per firm, sorted by time
    firm_events = [[] for _ in range(Nf)]
    for k in range(f2e.size(1)):
        u = int(f2e[0, k]); v = int(f2e[1, k])
        firm_events[u].append(v)
    for u in range(Nf):
        if firm_events[u]:
            evs = torch.tensor(firm_events[u], device=dev)
            firm_events[u] = evs[torch.argsort(day[evs])]        # ascending by time
        else:
            firm_events[u] = torch.empty(0, dtype=torch.long, device=dev)

    # --- firm -> similar firms (small out-degree expected)
    f2f = data[('firm','similar','firm')].edge_index.to(dev)
    sim_out = [[] for _ in range(Nf)]
    for k in range(f2f.size(1)):
        a = int(f2f[0, k]); b = int(f2f[1, k])
        sim_out[a].append(b)
        if treat_sim_undirected:
            sim_out[b].append(a)

    # --- helpers
    def last_k_prior(sorted_evs: torch.Tensor, t: int, K: int | None):
        """Return indices in sorted_evs with day < t; take last K if K is not None."""
        if sorted_evs.numel() == 0:
            return sorted_evs
        j = int(torch.searchsorted(day[sorted_evs], torch.tensor(t, device=dev), right=True))
        if j == 0:  # no prior events
            return sorted_evs[:0]
        if K is None:
            return sorted_evs[:j]
        j0 = max(0, j - K)
        return sorted_evs[j0:j]

    # --- containers
    same_src, same_dst, same_w = [], [], []
    peer_src, peer_dst, peer_w = [], [], []

    g_same = math.log(2.0)/half_life_same if use_time_decay else 0.0
    g_peer = math.log(2.0)/half_life_peer if use_time_decay else 0.0

    # --- main loop over all events as destinations
    for i in range(Ne):
        f = int(firm_of_event[i].item())
        if f < 0:
            # no firm link -> skip; event gets no same/peer edges (as requested)
            continue
        t_i = int(day[i].item())

        # SAME-FIRM: previous events of firm f
        evs_f = firm_events[f]
        prev = last_k_prior(evs_f, t_i, K_same)
        if prev.numel():
            same_src.append(prev)
            same_dst.append(torch.full_like(prev, i))
            if use_time_decay:
                Δ = (day[i] - day[prev]).float().clamp_min_(0)
                same_w.append(torch.exp(-g_same * Δ))
            else:
                same_w.append(torch.ones(prev.numel(), device=dev))

        # PEER-FIRM: previous events of each similar firm g
        for g in sim_out[f]:
            evs_g = firm_events[g]
            prev_g = last_k_prior(evs_g, t_i, K_peer)
            if prev_g.numel():
                peer_src.append(prev_g)
                peer_dst.append(torch.full_like(prev_g, i))
                if use_time_decay:
                    Δg = (day[i] - day[prev_g]).float().clamp_min_(0)
                    peer_w.append(torch.exp(-g_peer * Δg))
                else:
                    peer_w.append(torch.ones(prev_g.numel(), device=dev))

    # --- stitch tensors
    def _cat_long(parts):
        return torch.cat(parts) if parts else torch.empty(0, dtype=torch.long, device=dev)
    def _cat_float(parts):
        return torch.cat(parts) if parts else torch.empty(0, dtype=torch.float32, device=dev)

    s_src = _cat_long(same_src); s_dst = _cat_long(same_dst); s_w = _cat_float(same_w)
    p_src = _cat_long(peer_src); p_dst = _cat_long(peer_dst); p_w = _cat_float(peer_w)

    # --- write new relations
    data[('event','prev_same','event')].edge_index = torch.stack([s_src, s_dst], 0)
    data[('event','prev_same','event')].edge_attr  = s_w.unsqueeze(1) if s_w.numel() else None
    data[('event','prev_peer','event')].edge_index = torch.stack([p_src, p_dst], 0)
    data[('event','prev_peer','event')].edge_attr  = p_w.unsqueeze(1) if p_w.numel() else None

    # --- sanity: causality (only check if non-empty)
    if s_src.numel():
        assert (day[s_src] < day[s_dst]).all(), "Non-causal same-firm edge found"
    if p_src.numel():
        assert (day[p_src] < day[p_dst]).all(), "Non-causal peer edge found"

    return data
