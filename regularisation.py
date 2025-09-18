import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv
from torch_geometric.nn.conv import MessagePassing

# ------------------------ Weighted neighbor-only conv -------------------------
class WeightedAggConv(MessagePassing):
    """
    Mean aggregation of neighbor messages ONLY (no self-loop), with optional
    scalar edge weights taken from edge_attr[:, weight_col].
    Works for homo and hetero (x can be Tensor or (x_src, x_dst) tuple).
    """
    def __init__(self, in_channels_src, out_channels_dst, weight_col: int = 0):
        super().__init__(aggr='mean')
        self.lin_src = nn.Linear(in_channels_src, out_channels_dst)
        self.weight_col = weight_col

    def forward(self, x, edge_index, edge_attr=None, size=None):
        x_src, x_dst = x if isinstance(x, (tuple, list)) else (x, x)
        h_src = self.lin_src(x_src)                          # project source only
        return self.propagate(edge_index, x=(h_src, x_dst), edge_attr=edge_attr, size=size)

    def message(self, x_j, edge_attr):
        if edge_attr is not None:
            # use first column as scalar weight if present
            w = edge_attr[:, self.weight_col:self.weight_col+1]
            return w * x_j
        return x_j


# --------------------------- Gating (self vs neighbors) -----------------------
class SelfNeighborGate(nn.Module):
    """
    Compute alpha in (0,1) from self embedding; mix: h = alpha*h_self + (1-alpha)*h_nei.
    By default, the gate sees a stop-gradient self to avoid trivial solutions.
    """
    def __init__(self, d, hidden=64, detach_self=True):
        super().__init__()
        self.detached = detach_self
        self.g = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, h_self, h_nei):
        gate_in = h_self.detach() if self.detached else h_self
        alpha = torch.sigmoid(self.g(gate_in))               # (B,1)
        h = alpha * h_self + (1.0 - alpha) * h_nei
        return h, alpha


# ------------------------------- The main model -------------------------------
class EventFirmGatedGNN(nn.Module):
    """
    - Encodes event and firm features with MLPs (self paths).
    - Neighbor-only HeteroConv over:
        ('firm','rev_of','event')      -> messages into event from its firm(s)
        ('event','of','firm')          -> messages into firm from its events
        ('firm','similar','firm')      -> messages into firm from similar firms
    - Gate on event nodes blends self vs neighbor paths.
    - Predict scalar on event nodes (use your loss of choice).
    """
    def __init__(self, d_event_in, d_firm_in, d_hidden=128,
                 use_event_mask: bool = False, dropout=0.1, edge_weight_col=0):
        super().__init__()
        self.use_event_mask = use_event_mask
        # --- encoders (self paths)
        self.ev_enc = nn.Sequential(
            nn.Linear(d_event_in, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fm_enc = nn.Sequential(
            nn.Linear(d_firm_in, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Dropout(dropout),
        )

        # --- neighbor-only convs per relation
        self.nei_conv = HeteroConv({
            ('firm','rev_of','event'):  WeightedAggConv(d_hidden, d_hidden, weight_col=edge_weight_col),
            ('event','of','firm'):      WeightedAggConv(d_hidden, d_hidden, weight_col=edge_weight_col),
            ('firm','similar','firm'):  WeightedAggConv(d_hidden, d_hidden, weight_col=edge_weight_col),
        }, aggr='sum')  # sum across relations per target type

        self.ev_norm = nn.LayerNorm(d_hidden)
        self.fm_norm = nn.LayerNorm(d_hidden)
        self.gate    = SelfNeighborGate(d_hidden, hidden=max(32, d_hidden//4), detach_self=True)

        # prediction head on events
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(d_hidden, 1))

    def forward(self, data):
        # --------------- self encodings ----------------
        x_e = data['event'].x
        if self.use_event_mask and hasattr(data['event'], 'x_mask'):
            # concatenate boolean mask as features
            x_e = torch.cat([x_e, data['event'].x_mask.float()], dim=-1)
        h_self_e = self.ev_enc(x_e)                    # [Ne, d]
        h_self_f = self.fm_enc(data['firm'].x)         # [Nf, d]

        # --------------- neighbor messages ------------
        x_dict = {'event': h_self_e, 'firm': h_self_f}
        edge_index_dict = data.edge_index_dict

        # optional per-relation edge_attr dict (corr weights, recency, etc.)
        edge_attr_dict = {}
        for rel in [('firm','rev_of','event'), ('event','of','firm'), ('firm','similar','firm')]:
            store = data[rel]
            if hasattr(store, 'edge_attr') and store.edge_attr is not None:
                edge_attr_dict[rel] = store.edge_attr

        h_nei_dict = self.nei_conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)

        # normalize neighbor features
        h_nei_e = self.ev_norm(h_nei_dict['event'])
        h_nei_f = self.fm_norm(h_nei_dict['firm'])

        # --------------- gate at EVENT nodes ----------
        h_mix_e, alpha = self.gate(h_self_e, h_nei_e)  # alpha in (0,1)

        # predict ONLY on seed event nodes if batch sampler is used
        if hasattr(data['event'], 'batch_size'):
            out = self.head(h_mix_e[:data['event'].batch_size]).squeeze(-1)
            alpha_seed = alpha[:data['event'].batch_size]
        else:
            out = self.head(h_mix_e).squeeze(-1)
            alpha_seed = alpha

        # return everything needed for regularisers
        return {
            'y_hat_event': out,           # predictions on seed events if batch_size present
            'alpha_event': alpha_seed,    # gate on the same seeds
            'h_event_all': h_mix_e,       # full event embeddings (for regs if needed)
            'h_firm_all':  (h_self_f + h_nei_f),  # combined firm embeddings for regs
        }


# ------------------------------ Regularisers ---------------------------------
@torch.no_grad()
def _edge_weight_or_one(store):
    if hasattr(store, 'edge_attr') and store.edge_attr is not None and store.edge_attr.dim() >= 2:
        return store.edge_attr[:, 0].clamp_min(0.0)  # use first column as non-negative weight
    return None

def laplacian_reg_firm_sim(data, h_firm, lam=1e-3):
    """Laplacian smoothness on ('firm','similar','firm') using firm embeddings."""
    s, d = data[('firm','similar','firm')].edge_index
    w = _edge_weight_or_one(data[('firm','similar','firm')])
    diff2 = (h_firm[s] - h_firm[d]).pow(2).sum(dim=-1)
    if w is not None:
        diff2 = diff2 * w
    return lam * diff2.mean()

def event_firm_align_reg(data, h_event, h_firm, lam=5e-4):
    """Encourage event embedding to align with its firm's embedding along ('event','of','firm')."""
    s, d = data[('event','of','firm')].edge_index
    # if using seed-only supervision, it’s ok to align on all edges in the batch subgraph
    diff2 = (h_event[s] - h_firm[d]).pow(2).sum(dim=-1)
    return lam * diff2.mean()

def gate_penalty(alpha_seed, lam=1e-3):
    """Discourage alpha→1 (pure self path)."""
    return lam * alpha_seed.mean()



# Example dims (adjust if you concatenate masks)
d_event_in = data['event'].x.size(-1) + (data['event'].x_mask.size(-1) if hasattr(data['event'],'x_mask') else 0)
d_firm_in  = data['firm'].x.size(-1)

model = EventFirmGatedGNN(d_event_in=d_event_in, d_firm_in=d_firm_in, d_hidden=128,
                          use_event_mask=hasattr(data['event'], 'x_mask'), dropout=0.2).to(device)

# Higher LR on neighbor convs (as you tried)
conv_names = ('nei_conv.',)
param_groups = [
    {'params': [p for n,p in model.named_parameters() if any(k in n for k in conv_names)], 'lr': 3e-3},
    {'params': [p for n,p in model.named_parameters() if not any(k in n for k in conv_names)], 'lr': 1e-3}
]
opt = torch.optim.AdamW(param_groups, weight_decay=5e-4)

# Your loss (e.g., Laplace NLL, Huber, Quantile, or Censored version)
def huber(y, yhat, delta=0.2):
    r = (yhat - y).abs()
    return torch.where(r <= delta, 0.5*r*r/delta, r - 0.5*delta).mean()

model.train()
for batch in train_loader:      # your NeighborLoader with num_neighbors as specified
    batch = batch.to(device)
    out = model(batch)
    B = batch['event'].batch_size
    y_true = batch['event'].y[:B].view(-1).to(out['y_hat_event'].dtype)

    base = huber(y_true, out['y_hat_event'], delta=0.2)
    reg1 = laplacian_reg_firm_sim(batch, out['h_firm_all'], lam=1e-3)
    reg2 = event_firm_align_reg(batch, out['h_event_all'], out['h_firm_all'], lam=5e-4)
    reg3 = gate_penalty(out['alpha_event'], lam=1e-3)

    loss = base + reg1 + reg2 + reg3
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()


# Example dims (adjust if you concatenate masks)
d_event_in = data['event'].x.size(-1) + (data['event'].x_mask.size(-1) if hasattr(data['event'],'x_mask') else 0)
d_firm_in  = data['firm'].x.size(-1)

model = EventFirmGatedGNN(d_event_in=d_event_in, d_firm_in=d_firm_in, d_hidden=128,
                          use_event_mask=hasattr(data['event'], 'x_mask'), dropout=0.2).to(device)

# Higher LR on neighbor convs (as you tried)
conv_names = ('nei_conv.',)
param_groups = [
    {'params': [p for n,p in model.named_parameters() if any(k in n for k in conv_names)], 'lr': 3e-3},
    {'params': [p for n,p in model.named_parameters() if not any(k in n for k in conv_names)], 'lr': 1e-3}
]
opt = torch.optim.AdamW(param_groups, weight_decay=5e-4)

# Your loss (e.g., Laplace NLL, Huber, Quantile, or Censored version)
def huber(y, yhat, delta=0.2):
    r = (yhat - y).abs()
    return torch.where(r <= delta, 0.5*r*r/delta, r - 0.5*delta).mean()

model.train()
for batch in train_loader:      # your NeighborLoader with num_neighbors as specified
    batch = batch.to(device)
    out = model(batch)
    B = batch['event'].batch_size
    y_true = batch['event'].y[:B].view(-1).to(out['y_hat_event'].dtype)

    base = huber(y_true, out['y_hat_event'], delta=0.2)
    reg1 = laplacian_reg_firm_sim(batch, out['h_firm_all'], lam=1e-3)
    reg2 = event_firm_align_reg(batch, out['h_event_all'], out['h_firm_all'], lam=5e-4)
    reg3 = gate_penalty(out['alpha_event'], lam=1e-3)

    loss = base + reg1 + reg2 + reg3
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()



def center_event_dropout(batch, p=0.3):
    if not hasattr(batch['event'], 'batch_size'):
        return batch
    B = batch['event'].batch_size
    m = (torch.rand(B, device=batch['event'].x.device) < p)
    batch['event'].x[:B][m] = 0.0
    if hasattr(batch['event'], 'x_mask'):
        batch['event'].x_mask[:B][m] = 0.0
    return batch

# Use as a transform in your train NeighborLoader:
# train_loader = NeighborLoader(..., transform=lambda b: center_event_dropout(b, p=0.3))
