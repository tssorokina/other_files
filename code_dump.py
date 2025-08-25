import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import torch

def visualize_batch(batch, save_png=None, save_graphml=None):
    """
    Visualize a NeighborLoader hetero batch (prefer batch_size=1).
    - Events placed by their time on x-axis (requires batch['event'].time).
    - Firms placed in a separate row.
    - Seed event highlighted (first 'batch_size' events).
    - Edge styles by relation (solid/dashed/dotted/dashdot).
    """
    G = nx.DiGraph()

    # --- add nodes with attributes ---
    # Use tuple keys: (type, local_id)
    # Keep original global id for reference: n_id
    # For events: also store time and whether it's a seed.
    bsz = int(batch['event'].batch_size)
    event_times = batch['event'].time.cpu().numpy()
    event_nids  = batch['event'].n_id.cpu().numpy()
    firm_nids   = batch['firm'].n_id.cpu().numpy()

    for i in range(batch['event'].num_nodes):
        G.add_node(('event', i),
                   type='event',
                   global_id=int(event_nids[i]),
                   time=int(event_times[i]),
                   seed=bool(i < bsz))

    for i in range(batch['firm'].num_nodes):
        G.add_node(('firm', i),
                   type='firm',
                   global_id=int(firm_nids[i]))

    # --- add edges with 'rel' attribute ---
    def add_edges(et):
        e = batch.edge_index_dict.get(et)
        if e is None or e.numel() == 0:
            return
        src_t, rel, dst_t = et
        s = e[0].cpu().numpy()
        d = e[1].cpu().numpy()
        for si, di in zip(s, d):
            G.add_edge((src_t, int(si)), (dst_t, int(di)), rel=rel)

    for et in batch.edge_index_dict.keys():
        add_edges(et)

    # --- compute positions: time on x for events, firms stacked above ---
    # Events on y=0, x = normalized time rank
    if len(G) == 0:
        print("Empty graph")
        return

    times = np.array([G.nodes[n].get('time', None) for n in G.nodes if G.nodes[n]['type']=='event'])
    if times.size == 0:
        # fallback to spring layout
        pos = nx.spring_layout(G, seed=0)
    else:
        u_times = np.unique(times)
        t2x = {t: i for i, t in enumerate(np.sort(u_times))}
        pos = {}
        # events
        for n in G.nodes:
            if G.nodes[n]['type'] == 'event':
                x = t2x[G.nodes[n]['time']]
                pos[n] = np.array([x, 0.0], dtype=float)
        # firms: place at y=1, x = avg of connected events (if any), else spread by index
        firm_nodes = [n for n in G.nodes if G.nodes[n]['type']=='firm']
        for idx, n in enumerate(firm_nodes):
            nbr_events = [m for m in G.neighbors(n) if G.nodes[m]['type']=='event'] + \
                         [m for m in G.predecessors(n) if G.nodes[m]['type']=='event']
            if nbr_events:
                xs = [pos[m][0] for m in nbr_events if m in pos]
                x = float(np.mean(xs)) if xs else float(idx)
            else:
                x = float(idx)
            pos[n] = np.array([x, 1.0], dtype=float)

    # --- draw ---
    fig, ax = plt.subplots(figsize=(10, 5))
    # nodes
    ev_nodes = [n for n in G.nodes if G.nodes[n]['type']=='event']
    fm_nodes = [n for n in G.nodes if G.nodes[n]['type']=='firm']
    # seed events bigger marker
    ev_seed = [n for n in ev_nodes if G.nodes[n].get('seed', False)]
    ev_rest = [n for n in ev_nodes if n not in ev_seed]

    nx.draw_networkx_nodes(G, pos, nodelist=fm_nodes, node_shape='s', node_size=200, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=ev_rest, node_shape='o', node_size=100, ax=ax)
    if ev_seed:
        nx.draw_networkx_nodes(G, pos, nodelist=ev_seed, node_shape='o', node_size=300, ax=ax)

    # edges by relation with different linestyles (no colors)
    rel_styles = {
        'has': 'solid',
        'of': 'dashed',
        'similar': 'dashdot',
        'past_of': 'dotted',
    }
    for rel in sorted(set(nx.get_edge_attributes(G, 'rel').values())):
        es = [(u, v) for (u, v, r) in G.edges(data='rel') if r == rel]
        nx.draw_networkx_edges(G, pos, edgelist=es, style=rel_styles.get(rel, 'solid'), alpha=0.8, ax=ax)

    # minimal labels: seed event id + its time
    labels = {}
    for n in ev_seed:
        labels[n] = f"seed(e:{G.nodes[n]['global_id']}, t={G.nodes[n]['time']})"
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, ax=ax)

    # pretty axes
    ax.set_axis_off()
    ax.set_title("Batch subgraph (firms on top, events by time on x)")
    fig.tight_layout()

    if save_png:
        fig.savefig(save_png, dpi=150, bbox_inches='tight')
    if save_graphml:
        # export with stringified node keys for Gephi
        H = nx.relabel_nodes(G, lambda n: f"{n[0]}::{n[1]}")
        nx.write_graphml(H, save_graphml)
    return fig, ax, G
