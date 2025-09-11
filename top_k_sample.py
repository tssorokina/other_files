def topk_e2f_in_batch(batch, edge_type=('event','of','firm'), k=5, event_day_attr='day_idx'):
    store = batch[edge_type]
    s, d = store.edge_index
    e_day = batch['event'][event_day_attr][s]    # event day per edge

    # Sort within batch: per firm d, by e_day desc
    idx1 = torch.argsort(e_day, descending=True, stable=True)
    s1, d1 = s[idx1], d[idx1]
    idx2 = torch.argsort(d1, stable=True)
    s2, d2 = s1[idx2], d1[idx2]
    map_back = idx1[idx2]

    num_firms = batch['firm'].num_nodes
    counts = torch.bincount(d2, minlength=num_firms)
    starts = torch.cat([torch.zeros(1, dtype=torch.long, device=s.device),
                        counts.cumsum(0)[:-1]])
    pos = torch.arange(s2.numel(), device=s.device) - starts[d2]
    mask = (pos < k)

    keep = map_back[mask]
    store.edge_index = torch.stack([s[keep], d[keep]], dim=0)
    if store.edge_attr is not None:
        store.edge_attr = store.edge_attr[keep]
    return batch

train_loader = NeighborLoader(
    data,
    input_nodes=('event', data['event'].train_mask),
    num_neighbors={('event','of','firm'): [-1],  # get ALL, prune in transform
                   ('event','past_of','event'): [20, 10]},
    shuffle=True, batch_size=1024, transform=lambda b: topk_e2f_in_batch(b, k=5)
)
