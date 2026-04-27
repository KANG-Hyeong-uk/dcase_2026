import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self, label_smoothing=0.01):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits, labels, z=None, top_labels=None):
        return self.cross_entropy(logits, labels), {
            'ce': self.cross_entropy(logits, labels).detach(),
            'top': torch.tensor(0.0, device=logits.device),
            'contr': torch.tensor(0.0, device=logits.device),
        }


class HierarchicalLoss(nn.Module):
    """
    L_total = L_CE(2nd) + lambda_top * L_Top + lambda_contr * L_Contr

    L_Top: differentiable surrogate for 1[top(argmax(logits)) != top_label].
    Aggregates fine-class probabilities to top-class via the subclass->topclass
    membership matrix, then CE against top labels.

    L_Contr: supervised contrastive loss (Khosla et al., 2020) using top_class as
    the positive grouping. Operates on L2-normalized latent z.

    Reference: Anastasopoulou et al., DCASE Workshop 2025.
    """

    def __init__(self, subclass_to_topclass_tensor, num_top_classes,
                 lambda_top=0.3, lambda_contr=0.1, tau=0.07,
                 label_smoothing=0.01):
        super().__init__()
        sub2top = subclass_to_topclass_tensor.long()
        num_sub = int(sub2top.numel())
        num_top = int(num_top_classes)

        mask = torch.zeros(num_top, num_sub)
        for sub_id in range(num_sub):
            mask[sub2top[sub_id].item(), sub_id] = 1.0
        self.register_buffer('top_mask', mask)
        self.register_buffer('sub2top', sub2top)

        self.num_top = num_top
        self.num_sub = num_sub
        self.lambda_top = float(lambda_top)
        self.lambda_contr = float(lambda_contr)
        self.tau = float(tau)
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits, labels, z=None, top_labels=None):
        l_ce = self.ce(logits, labels)

        if top_labels is not None:
            log_p_sub = F.log_softmax(logits, dim=1)
            p_sub = log_p_sub.exp()
            p_top = p_sub @ self.top_mask.t()
            log_p_top = (p_top + 1e-12).log()
            l_top = F.nll_loss(log_p_top, top_labels.long())
        else:
            l_top = torch.zeros((), device=logits.device)

        l_contr = torch.zeros((), device=logits.device)
        if (z is not None) and (top_labels is not None) and (z.size(0) > 1):
            z_n = F.normalize(z, dim=1)
            sim = (z_n @ z_n.t()) / self.tau
            B = z.size(0)
            self_mask = torch.eye(B, dtype=torch.bool, device=z.device)

            sim_max = sim.masked_fill(self_mask, float('-inf')).max(dim=1, keepdim=True).values
            sim = sim - sim_max.detach()

            exp_sim = sim.exp().masked_fill(self_mask, 0.0)
            log_denom = exp_sim.sum(dim=1, keepdim=True).clamp(min=1e-12).log()
            log_prob = sim - log_denom

            pos_mask = (top_labels.unsqueeze(0) == top_labels.unsqueeze(1)) & (~self_mask)
            pos_count = pos_mask.sum(dim=1).clamp(min=1).float()
            mean_log_prob_pos = (log_prob * pos_mask.float()).sum(dim=1) / pos_count

            valid = pos_mask.any(dim=1)
            if valid.any():
                l_contr = -mean_log_prob_pos[valid].mean()

        total = l_ce + self.lambda_top * l_top + self.lambda_contr * l_contr
        return total, {
            'ce': l_ce.detach(),
            'top': l_top.detach() if isinstance(l_top, torch.Tensor) else torch.tensor(0.0, device=logits.device),
            'contr': l_contr.detach() if isinstance(l_contr, torch.Tensor) else torch.tensor(0.0, device=logits.device),
        }
