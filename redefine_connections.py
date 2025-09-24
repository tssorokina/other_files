import math, torch
from torch_scatter import scatter_add

def _get_event_firm_map(batch):
    """Return LongTensor firm_of_event [Ne] (assumes each event has a single firm)."""
    Ne = batch['event'].num_nodes
    firm_of_event = torch.full((Ne,), -1, dtype=torch.long, device=batch['event'].x.device)

    # Use ('event','of','firm') if present
    if ('event','of','firm') in batch.edge_types:
        s, d = batch[('event','of','firm')].edge_index
        firm_of_event[s] = d

    # Fill via reverse if any remain -1
    if ('firm','rev_of','event') in batch.edge_types:
        s, d = batch[('firm','rev_of','event')].edge_index
        # some events may appear multiple times; assume consistent firm id
        firm_of_event[d] = torch.where(firm_of_event[d] >= 0, firm_of_event[d], s)

    assert (firm_of_event >= 0).all(), "Some events missing firm mapping in the sampled subgraph."
    return firm_of_event

def collapse_to_event_only(
    batch,
    K_same=3,
    K_peer=3,
    half_life_same=180.0,   # trading days
    half_life_peer=90.0,
    sim_weight_col=0        # which col in firm-sim edge_attr holds similarity (corr) weight
):
    """
    Build two event->event relations in the CURRENT sampled batch:
      1) prev_same: last K_same past events of the seed's own firm
      2) prev_peer: last K_peer past events of each similar firm
    For multiple seeds (NeighborLoader batch), each seed gets its own star.
    """
    dev = batch['event'].x.device
    B   = int(batch['event'].batch_size)       # number of seeds in this subgraph
    seed_ids  = torch.arange(B, device=dev, dtype=torch.long)
    t_event   = batch['event'].time.long()     # trading-day index per event
    firm_of_e = _get_event_firm_map(batch)     # [Ne]

    # --- firm->similar firms adjacency present in the sampled batch
    if ('firm','similar','firm') in batch.edge_types:
        f2f_src, f2f_dst = batch[('firm','similar','firm')].edge_index
        if hasattr(batch[('firm','similar','firm')], 'edge_attr') and \
           batch[('firm','similar','firm')].edge_attr is not None:
            sim_w_all = batch[('firm','similar','firm')].edge_attr[:, sim_weight_col].float()
        else:
            sim_w_all = torch.ones(f2f_src.numel(), device=dev)
    else:
        # no peers in this batch → peer edges empty
        f2f_src = f2f_dst = sim_w_all = torch.empty(0, dtype=torch.long, device=dev)

    # Build quick index: for each firm f, which rows of (f2f_src->f2f_dst) start there?
    # We'll just scan per seed below (fanout is tiny).
    # Also build per-firm list of its events present in this batch, sorted by time.
    Ne = batch['event'].num_nodes
    order_e = torch.argsort(t_event)  # ascending time
    firms_sorted = firm_of_e[order_e]
    # CSR for firm -> events in time order
    F = int(batch['firm'].num_nodes)
    counts = torch.bincount(firms_sorted, minlength=F)
    rowptr = torch.zeros(F+1, dtype=torch.long, device=dev)
    rowptr[1:] = counts.cumsum(0)  # [F+1]

    gamma_same = math.log(2.0) / float(half_life_same)
    gamma_peer = math.log(2.0) / float(half_life_peer)

    src_same, dst_same, w_same = [], [], []
    src_peer, dst_peer, w_peer = [], [], []

    for seed in seed_ids.tolist():
        f0 = firm_of_e[seed].item()
        t0 = t_event[seed].item()

        # -------- same-firm: last K_same events of f0 strictly before t0
        s, e = rowptr[f0].item(), rowptr[f0+1].item()
        evs_f0 = order_e[s:e]                              # events of f0 sorted by time
        if e > s:
            # index of first > t0
            j = int(torch.searchsorted(t_event[evs_f0], torch.tensor(t0, device=dev), right=True))
            j0 = max(0, j - K_same)
            if j0 < j:
                cand = evs_f0[j0:j]
                Δ = (t0 - t_event[cand].float()).clamp_min_(0)
                rel = Δ - Δ.min()                          # nearest gets 0
                w   = torch.exp(-gamma_same * rel)
                src_same.append(cand)
                dst_same.append(torch.full_like(cand, seed))
                w_same.append(w)

        # -------- peer-firm: similar firms of f0 (directed as stored)
        if f2f_src.numel() > 0:
            mask = (f2f_src == f0)
            sim_neighbors = f2f_dst[mask]
            sim_w = sim_w_all[mask].clamp_min(0.0)
            if sim_neighbors.numel() > 0:
                for g, wg in zip(sim_neighbors.tolist(), sim_w.tolist()):
                    s2, e2 = rowptr[g].item(), rowptr[g+1].item()
                    if e2 <= s2: 
                        continue
                    evs_g = order_e[s2:e2]
                    j = int(torch.searchsorted(t_event[evs_g], torch.tensor(t0, device=dev), right=True))
                    j0 = max(0, j - K_peer)
                    if j0 < j:
                        cand = evs_g[j0:j]
                        Δ = (t0 - t_event[cand].float()).clamp_min_(0)
                        rel = Δ - Δ.min()
                        w   = float(wg) * torch.exp(-gamma_peer * rel)
                        src_peer.append(cand)
                        dst_peer.append(torch.full_like(cand, seed))
                        w_peer.append(w)

    def _stack(parts):
        if not parts:
            return (torch.empty(2,0, dtype=torch.long, device=dev),
                    torch.empty(0,   dtype=torch.float32, device=dev))
        s = torch.cat(parts[0]); d = torch.cat(parts[1]); w = torch.cat(parts[2]).float()
        return torch.stack([s, d], dim=0), w

    ei_same, ww_same = _stack((src_same, dst_same, w_same))
    ei_peer, ww_peer = _stack((src_peer, dst_peer, w_peer))

    # write into the batch as event-only relations
    batch[('event','prev_same','event')].edge_index = ei_same
    batch[('event','prev_same','event')].edge_attr  = ww_same.unsqueeze(1)  # [E,1]
    batch[('event','prev_peer','event')].edge_index = ei_peer
    batch[('event','prev_peer','event')].edge_attr  = ww_peer.unsqueeze(1)
    return batch
