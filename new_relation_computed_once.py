import torch

@torch.no_grad()
def build_event_only_relations_simple(
    data,
    time_attr: str = "time",
    treat_sim_undirected: bool = False,
):
    """
    Create two relations once on the full HeteroData:
      ('event','prev_same','event') : all prior events of the *same* firm  -> event
      ('event','prev_peer','event') : all prior events of *similar* firms  -> event

    - Uses only ('event','of','firm') and ('firm','similar','firm').
    - If an event has no ('event','of','firm') link, it gets *no* incoming edges.
    - No K-truncation, no edge weights.
    - Causality enforced: only sources with day < target day.

    Assumptions:
      - Each event has at most one firm in ('event','of','firm').
      - `data['event'].[time_attr]` is an integer day index.
    """
    dev = data['event'].x.device
    Ne  = data['event'].num_nodes
    Nf  = data['firm'].num_nodes

    # --- event times
    day = getattr(data['event'], time_attr).long().to(dev)  # [Ne]

    # --- event -> firm (may be missing for some events)
    e2f = data[('event','of','firm')].edge_index.to(dev)    # [2, E_e2f]
    firm_of_event = torch.full((Ne,), -1, dtype=torch.long, device=dev)
    # If there are multiple edges per event, last one wins; typically 1:1.
    firm_of_event[e2f[0]] = e2f[1]

    # --- build firm → events (derived from e2f), sorted by time
    firm_events = [[] for _ in range(Nf)]
    # we iterate over *all events* that have a firm
    ev_idx_with_firm = (firm_of_event >= 0).nonzero(as_tuple=False).view(-1)
    if ev_idx_with_firm.numel():
        # group by firm
        firms_for_ev = firm_of_event[ev_idx_with_firm]
        # sort by firm then time
        order = torch.argsort(firms_for_ev, stable=True)
        ev_sorted = ev_idx_with_firm[order]
        firms_sorted = firms_for_ev[order]
        # within each firm, sort by time
        time_order = torch.argsort(day[ev_sorted], stable=True)
        ev_sorted = ev_sorted[time_order]
        firms_sorted = firms_sorted[time_order]
        # split into lists
        # (Python lists are fine here; degrees are small, and we touch each event once)
        for f, ev in zip(firms_sorted.tolist(), ev_sorted.tolist()):
            firm_events[f].append(ev)
        # convert lists to tensors for searchsorted
        for f in range(Nf):
            if firm_events[f]:
                firm_events[f] = torch.tensor(firm_events[f], device=dev, dtype=torch.long)
            else:
                firm_events[f] = torch.empty(0, device=dev, dtype=torch.long)

    # --- firm → similar firms adjacency (from f2f); directed by default
    f2f = data[('firm','similar','firm')].edge_index.to(dev)  # [2, E_sim]
    firm_neighbors = [[] for _ in range(Nf)]
    for a, b in zip(f2f[0].tolist(), f2f[1].tolist()):
        firm_neighbors[a].append(b)
        if treat_sim_undirected:
            firm_neighbors[b].append(a)

    # --- build event→event edges
    same_src, same_dst = [], []
    peer_src, peer_dst = [], []

    # helper: take all prior events (< t) via searchsorted (firm_events[f] is time-sorted)
    def prior_events(ev_list: torch.Tensor, t: int):
        if ev_list.numel() == 0:
            return ev_list
        # day[ev_list] is sorted ascending
        j = int(torch.searchsorted(day[ev_list], torch.tensor(t, device=dev), right=True))
        return ev_list[:j]  # all strictly earlier

    for i in range(Ne):
        f = int(firm_of_event[i].item())
        if f < 0:
            continue  # skip events with no firm
        t_i = int(day[i].item())

        # same-firm: all prior events of firm f
        prev_same = prior_events(firm_events[f], t_i)
        if prev_same.numel():
            same_src.append(prev_same)
            same_dst.append(torch.full_like(prev_same, i))

        # peer firms: all prior events of each neighbor g != f
        for g in firm_neighbors[f]:
            if g == f:
                continue
            prev_peer = prior_events(firm_events[g], t_i)
            if prev_peer.numel():
                peer_src.append(prev_peer)
                peer_dst.append(torch.full_like(prev_peer, i))

    # stitch
    def _cat(parts, dtype=torch.long):
        return torch.cat(parts) if parts else torch.empty(0, dtype=dtype, device=dev)

    s_src = _cat(same_src); s_dst = _cat(same_dst)
    p_src = _cat(peer_src); p_dst = _cat(peer_dst)

    # write back (no edge_attr)
    data[('event','prev_same','event')].edge_index = torch.stack([s_src, s_dst], 0)
    data[('event','prev_same','event')].edge_attr  = None
    data[('event','prev_peer','event')].edge_index = torch.stack([p_src, p_dst], 0)
    data[('event','prev_peer','event')].edge_attr  = None

    # sanity: causality (if any edges)
    if s_src.numel():
        assert (day[s_src] < day[s_dst]).all(), "Non-causal edge in prev_same"
    if p_src.numel():
        assert (day[p_src] < day[p_dst]).all(), "Non-causal edge in prev_peer"

    return data
