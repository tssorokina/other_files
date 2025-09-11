def keep_topk_recent_event_edges(data, edge_type=('event','of','firm'),
                                 k=5, event_day_attr='day_idx'):
    """Keep only K most recent ('event'→'firm') edges per firm, in-place."""
    store = data[edge_type]
    src, dst = store.edge_index
    day = data['event'][event_day_attr]          # [num_event_nodes] int day
    e_day = day[src]                             # [E] day of each edge's event

    # Stable two-step sort to get groups (dst) with events sorted by recency:
    idx1 = torch.argsort(e_day, descending=True, stable=True)
    src1, dst1 = src[idx1], dst[idx1]
    idx2 = torch.argsort(dst1, stable=True)
    src2, dst2 = src1[idx2], dst1[idx2]
    keep_map = idx1[idx2]                        # mapping back to original edges

    # Compute rank within each firm block:
    E = src2.numel()
    num_firms = data['firm'].num_nodes
    counts = torch.bincount(dst2, minlength=num_firms)              # deg per firm
    starts = torch.cat([torch.zeros(1, dtype=torch.long, device=src.device),
                        counts.cumsum(0)[:-1]])                     # start idx per firm
    pos = torch.arange(E, device=src.device) - starts[dst2]         # 0,1,2,...
    mask = (pos < k)

    keep_edges = keep_map[mask]
    store.edge_index = torch.stack([src[keep_edges], dst[keep_edges]], dim=0)
    if store.edge_attr is not None:
        store.edge_attr = store.edge_attr[keep_edges]
    return data
