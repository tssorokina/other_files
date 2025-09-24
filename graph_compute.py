from torch_geometric.nn import global_mean_pool, global_max_pool

class EventFirmGatedGNN(nn.Module):
    # ... keep your __init__ as-is, but add:
    def __init__(self, d_event_in, d_firm_in, d_hidden=128,
                 use_event_mask: bool = False, dropout=0.1, edge_weight_col=0):
        super().__init__()
        # (unchanged encoders / conv / norms / gate / node head)
        # ...
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(d_hidden, 1))  # node head

        # --- NEW: graph head over pooled event+firm embeddings (mean+max) ---
        self.graph_head = nn.Sequential(
            nn.Linear(4*d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, 1)
        )

    def forward(self, data):
        # (unchanged) compute h_self_e, h_self_f, neighbor messages, gating:
        # ...
        h_mix_e, alpha = self.gate(h_self_e, h_nei_e)
        h_firm_all = (h_self_f + h_nei_f)

        # NODE-LEVEL OUTPUT on seeds (kept for backwards compatibility)
        if hasattr(data['event'], 'batch_size'):
            y_node = self.head(h_mix_e[:data['event'].batch_size]).squeeze(-1)
            alpha_seed = alpha[:data['event'].batch_size]
        else:
            y_node = self.head(h_mix_e).squeeze(-1)
            alpha_seed = alpha

        # --------- NEW: GRAPH-LEVEL READOUT (one scalar per seed graph) ---------
        if not hasattr(data['event'], 'graph_id'):
            # fall back to node-only if transform wasn't applied
            y_graph = y_node
        else:
            gid_e = data['event'].graph_id          # [Ne] in [0..B-1]
            gid_f = data['firm'].graph_id           # [Nf] in [0..B-1]
            B = int(data['event'].batch_size)

            # pool event and firm embeddings per graph
            e_mean = global_mean_pool(h_mix_e, gid_e, size=B)
            e_max  = global_max_pool(h_mix_e,  gid_e, size=B)
            f_mean = global_mean_pool(h_firm_all, gid_f, size=B)
            f_max  = global_max_pool(h_firm_all,  gid_f, size=B)

            g = torch.cat([e_mean, e_max, f_mean, f_max], dim=-1)  # [B, 4*d_hidden]
            y_graph = self.graph_head(g).squeeze(-1)               # [B]

        return {
            'y_hat_event': y_node,          # seed-wise node outputs
            'y_hat_graph': y_graph,         # graph outputs (one per seed)
            'alpha_event': alpha_seed,
            'h_event_all': h_mix_e,
            'h_firm_all':  h_firm_all,
        }
