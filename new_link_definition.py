import math, torch

def make_event_only_star(
    batch,
    K_same: int = 3,          # last K of same firm
    K_peer: int = 2,          # last K of each similar firm
    half_life_days: float = 90.0,   # time-decay half-life
    time_attr: str = 'time'   # int trading-day index on event nodes
):
    device = batch['event'].x.device
    B = int(batch['event'].batch_size)
    Ne = batch['event'].num_nodes
    Nf = batch['firm'].num_nodes
    day = batch['event'].__getattr__(time_attr).long()  # [Ne]
    gamma = math.log(2.0) / float(half_life_days)

    # Map event -> firm for nodes present in the subgraph
    e2f = batch[('event','of','firm')].edge_index  # src=event, dst=firm
    firm_of_event = torch.full((Ne,), -1, dtype=torch.long, device=device)
    firm_of_event[e2f[0]] = e2f[1]                 # assume 1 firm per event

    # Build (firm_src -> firm_dst) similarity weight lookup from edges present in batch
    f2f = batch[('firm','similar','firm')].edge_index
    if hasattr(batch[('firm','similar','firm')], 'edge_attr') and \
       batch[('firm','similar','firm')].edge_attr is not None and \
       batch[('firm','similar','firm')].edge_attr.size(1) >= 1:
        sim_w = batch[('firm','similar','firm')].edge_attr[:, 0]
    else:
        sim_w = torch.ones(f2f.size(1), device=device)
    # store in a dict for quick access
    sim_dict = {}
    for k in range(f2f.size(1)):
        sim_dict[(int(f2f[0, k]), int(f2f[1, k]))] = float(sim_w[k].item())

    # Preindex events by firm (within this batch)
    # For speed we won't build CSR; we will boolean-filter per firm (degrees are small in ego graphs).
    src_same, dst_same, w_same = [], [], []
    src_peer, dst_peer, w_peer = [], [], []

    seed_ids = torch.arange(B, device=device)            # seed event node indices
    seed_firms = firm_of_event[seed_ids]
    seed_days  = day[seed_ids]

    for i in range(B):
        e_i = int(seed_ids[i].item())
        f_i = int(seed_firms[i].item())
        t_i = int(seed_days[i].item())

        # ---- SAME-FIRM: last K events of f_i with day < t_i ----
        mask_same = (firm_of_event == f_i) & (day < t_i)
        idx_same = mask_same.nonzero(as_tuple=False).view(-1)
        if idx_same.numel() > 0:
            # pick last K by time
            take = idx_same[torch.argsort(day[idx_same])][-min(K_same, idx_same.numel()):]
            Δ = (day[e_i] - day[take]).float().clamp_min_(0)
            w = torch.exp(-gamma * Δ)                    # time-decay; use as message weight
            src_same.append(take)
            dst_same.append(torch.full_like(take, e_i))
            w_same.append(w)

        # ---- PEER-FIRM: last K events of each neighbor g with day < t_i ----
        # collect outgoing similar firms from f_i
        # (assumes direction matters; if undirected, also check reverse)
        peer_firms = [dst for (src, dst) in sim_dict.keys() if src == f_i]
        for g in peer_firms:
            mask_peer = (firm_of_event == g) & (day < t_i)
            idx_peer = mask_peer.nonzero(as_tuple=False).view(-1)
            if idx_peer.numel() == 0:
                continue
            take = idx_peer[torch.argsort(day[idx_peer])][-min(K_peer, idx_peer.numel()):]
            Δ = (day[e_i] - day[take]).float().clamp_min_(0)
            w_time = torch.exp(-gamma * Δ)
            w_sim  = float(sim_dict[(f_i, g)])
            w = w_time * w_sim
            src_peer.append(take)
            dst_peer.append(torch.full_like(take, e_i))
            w_peer.append(w)

    # Stitch lists
    if len(src_same):
        src_s = torch.cat(src_same); dst_s = torch.cat(dst_same); w_s = torch.cat(w_same)
    else:
        src_s = torch.empty(0, dtype=torch.long, device=device)
        dst_s = torch.empty(0, dtype=torch.long, device=device)
        w_s   = torch.empty(0, dtype=torch.float32, device=device)

    if len(src_peer):
        src_p = torch.cat(src_peer); dst_p = torch.cat(dst_peer); w_p = torch.cat(w_peer)
    else:
        src_p = torch.empty(0, dtype=torch.long, device=device)
        dst_p = torch.empty(0, dtype=torch.long, device=device)
        w_p   = torch.empty(0, dtype=torch.float32, device=device)

    # Write new event→event relations into the batch (used only for compute)
    batch[('event','prev_same','event')].edge_index = torch.stack([src_s, dst_s], 0)
    batch[('event','prev_same','event')].edge_attr  = w_s.unsqueeze(1)  # first col = weight
    batch[('event','prev_peer','event')].edge_index = torch.stack([src_p, dst_p], 0)
    batch[('event','prev_peer','event')].edge_attr  = w_p.unsqueeze(1)

    return batch
