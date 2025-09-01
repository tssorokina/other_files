import torch
import torch.nn as nn

def init_embeddings_orthogonal(module: nn.Module, gain: float = 1.0, seed: int | None = None):
    """
    Walks the module tree and orthogonally inits every nn.Embedding.weight.
    If padding_idx is set, that row is zeroed after init.
    """
    if seed is not None:
        torch.manual_seed(seed)

    for name, m in module.named_modules():
        if isinstance(m, nn.Embedding):
            with torch.no_grad():
                nn.init.orthogonal_(m.weight, gain=gain)  # 2D matrix → orthogonal cols/rows
                if m.padding_idx is not None:
                    m.weight[m.padding_idx].zero_()


class FirmCatEncoderFromMatrix(nn.Module):
    def __init__(self, cardinalities, out_dim, dropout=0.1, use_padding=True, init_gain=1.0):
        super().__init__()
        embs, dims = [], []
        for C in cardinalities:
            d = int(min(64, max(8, round((C ** 0.5) * 2))))
            embs.append(nn.Embedding(C + (1 if use_padding else 0), d,
                                     padding_idx=0 if use_padding else None))
            dims.append(d)
        self.embs = nn.ModuleList(embs)
        self.proj = nn.Linear(sum(dims), out_dim)
        self.do = dropout

        # orthogonal init for all embedding tables
        init_embeddings_orthogonal(self, gain=init_gain)

        # (optional) init projection too
        nn.init.kaiming_uniform_(self.proj.weight, a=0.0, nonlinearity='relu')
        nn.init.zeros_(self.proj.bias)

    def forward(self, firm_x_long: torch.Tensor):
        parts = [emb(firm_x_long[:, i]) for i, emb in enumerate(self.embs)]
        z = torch.cat(parts, dim=-1) if parts else firm_x_long.new_zeros(firm_x_long.size(0), 0).float()
        z = nn.functional.dropout(z, p=self.do, training=self.training)
        return nn.functional.relu(self.proj(z))
