# Updated visualization: place seed firm slightly above seed event (no overlap)
# - Events: y=0
# - Firms : y=1
# - Seed event(s): y=seed_y
# - Seed firm(s) : y=seed_y + seed_gap
#
# Edge styles only (no colors), seeds labeled.

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def visualize_batch_centered_offset(batch, seed_y=0.5, seed_gap=0.12, save_png=None, save_graphml=None):
    """
    Visualize a hetero NeighborLoader batch with seeds centered between rows.
    The seed firm is drawn slightly ABOVE the seed event so they don't overlap.

    Parameters
    ----------
    batch : PyG HeteroData mini-batch
    seed_y : float, default 0.5
        Base Y-position for the seed event(s).
    seed_gap : float, default 0.12
        Vertical offset added to seed firm(s) relative to seed_y.
    save_png : str or None
        Optional path to save the PNG figure.
    save_graphml : str or None
        Optional path to export a GraphML for Gephi.
    """
    G = nx.DiGraph()

    # --- discover seeds ---
    bsz = int(getattr(batch['event'], 'batch_size', 1))
    seed_event_idx = set(range(bsz))

    seed_firm_idx = set()
    # firm -> event
    e = batch.edge_index_dict.get(('firm','has','event'))
    if e is not None and e.numel() > 0:
        dst = e[1].cpu().numpy()
        src = e[0].cpu().numpy()
        for se in seed_event_idx:
            seed_firm_idx.update(map(int, src[dst == se]))
    # event -> firm
    e = batch.edge_index_dict.get(('event','of','firm'))
    if e is not None and e.numel() > 0:
        src = e[0].cpu().numpy()
        dst = e[1].cpu().numpy()
        for se in seed_event_idx:
            seed_firm_idx.update(map(int, dst[src == se]))

    # --- add nodes ---
    event_times = batch['event'].time.cpu().numpy() if hasattr(batch['event'], 'time') else None
    event_nids  = batch['event'].n_id.cpu().numpy() if hasattr(batch['event'], 'n_id') else np.arange(batch['event'].num_nodes)
    firm_nids   = batch['firm'].n_id.cpu().numpy()  if hasattr(batch['firm'], 'n_id')  else np.arange(batch['firm'].num_nodes)

    for i in range(batch['event'].num_nodes):
        G.add_node(('event', i),
                   type='event',
                   global_id=int(event_nids[i]),
                   time=(int(event_times[i]) if event_times is not None else None),
                   seed=(i in seed_event_idx))

    for i in range(batch['firm'].num_nodes):
        G.add_node(('firm', i),
                   type='firm',
                   global_id=int(firm_nids[i]),
                   seed=(i in seed_firm_idx))

    # --- add edges ---
    for (src_t, rel, dst_t), eidx in batch.edge_index_dict.items():
        if eidx is None or eidx.numel() == 0:
            continue
        s = eidx[0].cpu().numpy()
        d = eidx[1].cpu().numpy()
        for si, di in zip(s, d):
            G.add_edge((src_t, int(si)), (dst_t, int(di)), rel=rel)

    if len(G) == 0:
        print("Empty graph")
        return None

    # --- positions ---
    pos = {}
    if event_times is not None:
        u_times = np.unique(event_times)
        t2x = {t: i for i, t in enumerate(np.sort(u_times))}
    else:
        t2x = None

    # events
    for n in G.nodes:
        if G.nodes[n]['type'] == 'event':
            x = float(t2x[G.nodes[n]['time']]) if t2x is not None else float(n[1])
            y = float(seed_y) if G.nodes[n]['seed'] else 0.0
            pos[n] = np.array([x, y], dtype=float)

    # firms
    firm_nodes = [n for n in G.nodes if G.nodes[n]['type']=='firm']
    for idx, n in enumerate(firm_nodes):
        nbr_events = [m for m in G.predecessors(n) if G.nodes[m]['type']=='event'] + \
                     [m for m in G.successors(n)   if G.nodes[m]['type']=='event']
        xs = [pos[m][0] for m in nbr_events if m in pos]
        x = float(np.mean(xs)) if xs else float(idx)
        y = (seed_y + seed_gap) if G.nodes[n]['seed'] else 1.0
        pos[n] = np.array([x, y], dtype=float)

    # --- draw ---
    fig, ax = plt.subplots(figsize=(10, 5))

    ev_nodes = [n for n in G.nodes if G.nodes[n]['type']=='event' and not G.nodes[n]['seed']]
    fm_nodes = [n for n in G.nodes if G.nodes[n]['type']=='firm'  and not G.nodes[n]['seed']]
    ev_seed  = [n for n in G.nodes if G.nodes[n]['type']=='event' and G.nodes[n]['seed']]
    fm_seed  = [n for n in G.nodes if G.nodes[n]['type']=='firm'  and G.nodes[n]['seed']]

    if fm_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=fm_nodes, node_shape='s', node_size=200, ax=ax)
    if ev_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=ev_nodes, node_shape='o', node_size=120, ax=ax)
    if fm_seed:
        nx.draw_networkx_nodes(G, pos, nodelist=fm_seed, node_shape='s', node_size=360, ax=ax)
    if ev_seed:
        nx.draw_networkx_nodes(G, pos, nodelist=ev_seed, node_shape='o', node_size=300, ax=ax)

    # edges with different linestyles per relation
    rel_styles = {'has':'solid', 'of':'dashed', 'similar':'dashdot', 'past_of':'dotted'}
    edge_attrs = nx.get_edge_attributes(G, 'rel')
    for rel in sorted(set(edge_attrs.values())):
        es = [(u, v) for (u, v), r in edge_attrs.items() if r == rel]
        nx.draw_networkx_edges(G, pos, edgelist=es, style=rel_styles.get(rel, 'solid'), alpha=0.85, ax=ax)

    # labels for seeds
    labels = {}
    for n in ev_seed:
        t = G.nodes[n].get('time', None)
        labels[n] = f"seed_event(id={G.nodes[n]['global_id']}" + (f", t={t}" if t is not None else "") + ")"
    for n in fm_seed:
        labels[n] = f"seed_firm(id={G.nodes[n]['global_id']})"
    if labels:
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, ax=ax)

    ax.set_axis_off()
    ax.set_title("Batch subgraph with seed event and seed firm centered (firm slightly above)")
    fig.tight_layout()

    if save_png:
        fig.savefig(save_png, dpi=150, bbox_inches='tight')
    if save_graphml:
        H = nx.relabel_nodes(G, lambda n: f"{n[0]}::{n[1]}")
        nx.write_graphml(H, save_graphml)

    return fig, ax, G
