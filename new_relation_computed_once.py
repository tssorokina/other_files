import math
import torch

@torch.no_grad()
def build_event_only_relations(
    data,
    time_attr: str = 'time',            # int/long trading-day index on event nodes
    use_time_decay: bool = True,        # attach exp(-gamma Δt) as weight
    half_life_same: float = 90.0,       # only used if use_time_decay
    half_life_peer: float = 60.0,       # only used if use_time_decay
    K_same: int | None = None,          # if None, use all prior events present
    K_peer: int | None = None           # if None, use all prior events present
):
    """
    Creates two new relations on `data`:
      ('event','prev_same','event') and ('event','prev_peer','event'),
    using only the *original* hetero edges.

    Assumptions:
      - Exactly 1 firm per event.
      - data[('firm','rev_of','event')] contains the subset of each firm's events
        you want to consider (already pruned, if desired).
      - data[('firm','similar','firm')] already contains exactly the similar firms you want.
    """

    device = data['event'].x.device
    day = getattr(data['event'], time_attr).long().to(device)     # [Ne]

    # --- event -> firm mapping (one firm per event expected)
    e2f = data[('event','of','firm')].edge_index.to(device)       # [2, E_e2f]
    assert e2f.size(1) >= data['event'].num_nodes, "missing event→firm links?"
    firm_of_event = torch.full((data['event'].num_nodes,), -1, dtype=torch.long, device=device)
    firm_of_event[e2f[0]] = e2f[1]
    assert (firm_of_event >= 0).all(), "some events have no firm"

    # --- firm -> its events present in the dataset (we'll use for 'same' edges)
    f2e = data[('firm','rev_of','event')].edge_index.to(device)   # [2, E_f2e], src=firm, dst=event
    # bucket firm events and sort by time
    num_firm = data['firm'].num_nodes
    firm_events = [[] for _ in range(num_firm)]
    for k in range(f2e.size(1)):
        u = int(f2e[0, k])
        v = int(f2e[1, k])
        firm_events[u].append(v)
    for u in range(num_firm):
        if firm_events[u]:
            evs = torch.tensor(firm_events[u], device=device)
            firm_events[u] = evs[torch.argsort(day[evs])]  # ascending time
        else:
            firm_events[u] = torch.empty(0, dtype=torch.long, device=device)

    # --- firm -> similar firm list (for 'peer' edges)
    f2f = data[('firm','similar','firm')].edge_index.to(device)   # [2, E_sim]
    # Build outgoing adjacency as Python lists for simplicity (degrees are small)
    sim_out = [[] for _ in range(num_firm)]
    for k in range(f2f.size(1)):
        a = int(f2f[0, k]); b = int(f2f[1, k])
        sim_out[a].append(b)

    # --- helpers
    def last_k_prior(sorted_evs: torch.Tensor, t: int, K: int | None):
        """Return indices in sorted_evs with day < t; take last K if K given."""
        if sorted_evs.numel() == 0:
            return sorted_evs
        # upper bound index j s.t. day[sorted_evs[:j]] < t
        j = int(torch.searchsorted(day[sorted_evs], torch.tensor(t, device=device), right=True))
        if j == 0:
            return sorted_evs[:0]
        if K is None:
            return sorted_evs[:j]
        j0 = max(0, j - K)
        return sorted_evs[j0:j]

    # --- containers for new edges
    same_src, same_dst, same_w = [], [], []
    peer_src, peer_dst, peer_w = [], [], []

    g_same = math.log(2.0)/half_life_same if use_time_decay else None
    g_peer = math.log(2.0)/half_life_peer if use_time_decay else None

    Ne = data['event'].num_nodes
    for i in range(Ne):
        f = int(firm_of_event[i].item())
        t = int(day[i].item())

        # SAME-FIRM: previous events of firm f
        evs_f = firm_events[f]
        prev = last_k_prior(evs_f, t, K_same)
        if prev.numel():
            same_src.append(prev)
            same_dst.append(torch.full_like(prev, i))
            if use_time_decay:
                Δ = (day[i] - day[prev]).float().clamp_min_(0)
                same_w.append(torch.exp(-g_same * Δ))
            else:
                same_w.append(torch.ones(prev.numel(), device=device))

        # PEER-FIRM: previous events of each similar firm g
        for g in sim_out[f]:
            evs_g = firm_events[g]
            prev_g = last_k_prior(evs_g, t, K_peer)
            if prev_g.numel():
                peer_src.append(prev_g)
                peer_dst.append(torch.full_like(prev_g, i))
                if use_time_decay:
                    Δg = (day[i] - day[prev_g]).float().clamp_min_(0)
                    peer_w.append(torch.exp(-g_peer * Δg))
                else:
                    peer_w.append(torch.ones(prev_g.numel(), device=device))

    # stitch
    def _stitch(parts):
        if parts:
            return torch.cat(parts)
        return torch.empty(0, dtype=torch.long, device=device)

    s_src = _stitch(same_src); s_dst = _stitch(same_dst)
    p_src = _stitch(peer_src); p_dst = _stitch(peer_dst)
    s_w   = torch.cat(same_w) if same_w else torch.empty(0, dtype=torch.float32, device=device)
    p_w   = torch.cat(peer_w) if peer_w else torch.empty(0, dtype=torch.float32, device=device)

    # write new relations
    data[('event','prev_same','event')].edge_index = torch.stack([s_src, s_dst], 0)
    data[('event','prev_same','event')].edge_attr  = s_w.unsqueeze(1) if s_w.numel() else None
    data[('event','prev_peer','event')].edge_index = torch.stack([p_src, p_dst], 0)
    data[('event','prev_peer','event')].edge_attr  = p_w.unsqueeze(1) if p_w.numel() else None

    # sanity: all sources strictly earlier than targets
    if s_src.numel():
        assert (day[s_src] < day[s_dst]).all()
    if p_src.numel():
        assert (day[p_src] < day[p_dst]).all()

    return data
