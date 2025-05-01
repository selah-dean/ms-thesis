import numpy as np
import torch
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight


class GraphCalibrationLoss(torch.nn.Module):
    def __init__(self, gamma=0.01):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        Args:
            logits: Tensor of shape [B, C]
            targets: LongTensor of shape [B] with class indices
        Returns:
            Scalar loss value
        """
        probs = F.softmax(logits, dim=-1)  # [B, C]
        log_probs = torch.log(probs + 1e-12)  # numerical stability

        targets_onehot = F.one_hot(
            targets, num_classes=logits.size(-1)
        ).float()  # [B, C]
        weight = 1 + self.gamma * probs  # [B, C]
        loss = -weight * targets_onehot * log_probs  # [B, C]
        return loss.sum(dim=1).mean()  # average over batch


def compute_class_weights(y):
    y_np = y.cpu().numpy()
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_np), y=y_np
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(y.device)
    return class_weights