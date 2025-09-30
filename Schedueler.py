import math, torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

model = model.to(device)
base_lr = 8e-5
optimizer = AdamW(model.parameters(), lr=base_lr, weight_decay=5e-4)

E = 40                                   # total epochs
steps_per_epoch = len(train_loader)
total_steps = E * steps_per_epoch
warmup_steps = max(100, int(0.1 * total_steps))   # 10% warmup
eta_min_ratio = 0.1                                # final LR = 0.1 * base_lr

# schedulers
warmup = LambdaLR(optimizer, lr_lambda=lambda s: min(1.0, s / warmup_steps))
cosine = CosineAnnealingLR(
    optimizer,
    T_max=total_steps - warmup_steps,
    eta_min=base_lr * eta_min_ratio,
)
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

# optional SWA in last few epochs
use_swa, swa_epochs = True, 6
if use_swa:
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=base_lr * 0.5)

global_step = 0
for epoch in range(E):
    model.train()
    for batch in train_loader:
        global_step += 1
        loss = train_step(batch)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()                    # step PER BATCH

    # swap to SWA schedule for the tail epochs
    if use_swa and epoch >= E - swa_epochs:
        swa_scheduler.step()

    # val …
# finalize SWA
if use_swa:
    swa_model.update_parameters(model)
    update_bn(train_loader, swa_model, device=device)  # one pass to calibrate BN
    model = swa_model
