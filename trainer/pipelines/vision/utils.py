# utils.py
import inspect, math, torch, torch.nn as nn
import matplotlib.pyplot as plt


def _infer_cnn_in_channels(model):
    # Try to read the first Conv2d’s in_channels if available
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            return int(m.in_channels)
    return None  # fall back to heuristic

def shape_batch_for_model(model, x):
    """
    x: torch.Tensor of shape (B, F) or (B, C, H, W) or (F) or (C, H, W)
    Returns a tensor shaped appropriately for the given model.
    - If model ctor has `input_dim`: flatten to (B, F)
    - If model ctor has `in_channels`: ensure (B, C, H, W)
      * If flat, try to unflatten using known/heuristic C and square H=W
    """
    if x.dim() == 3:  # (C,H,W) single sample -> make batched
        x = x.unsqueeze(0)
    if x.dim() == 1:  # (F) single sample -> make batched
        x = x.unsqueeze(0)

    ctor_params = set(inspect.signature(model.__class__).parameters.keys())
    is_mlp = "input_dim" in ctor_params
    is_cnn = "in_channels" in ctor_params

    if is_mlp:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return x

    if is_cnn:
        if x.dim() == 2:
            # Try to unflatten
            F = x.size(1)
            C_known = _infer_cnn_in_channels(model)  # e.g., 3 for CIFAR10 SimpleCNN
            candidates = [C_known] if C_known else [3, 1]
            for C in [c for c in candidates if c]:
                if F % C == 0:
                    HW = F // C
                    H = int(math.isqrt(HW))
                    if H * H == HW:
                        return x.view(x.size(0), C, H, H)
            raise RuntimeError(
                f"Flat batch {tuple(x.shape)} for CNN and could not infer (C,H,W). "
                "Check that the dataset is not being flattened downstream."
            )
        # Already images
        if x.dim() != 4:
            raise RuntimeError(f"Expected (B,C,H,W) for CNN, got {tuple(x.shape)}")
        return x

    # Fallback if the model constructor doesn’t advertise either param
    return x if x.dim() > 2 else x.view(x.size(0), -1)

