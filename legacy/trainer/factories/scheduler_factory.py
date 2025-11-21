import math
import torch
import torch.optim as optim
from trainer.constants_models import SCHEDULER_SPECS, get_scheduler_spec


class CosineAnnealingWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, eta_min=0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step < self.warmup_steps:
            return [base_lr * step / self.warmup_steps for base_lr in self.base_lrs]
        else:
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return [
                self.eta_min + (base_lr - self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


class SchedulerFactory:
    @staticmethod
    def create(scheduler_name, optimizer, param_overrides=None):
        spec = get_scheduler_spec(scheduler_name)
        if not spec.get("enabled", False):
            return None

        params = spec.get("params", {}).copy()
        if param_overrides:
            params.update(param_overrides)

        if spec.get("class") == "custom" or scheduler_name == "cosine_warmup":
            return CosineAnnealingWarmupScheduler(
                optimizer,
                warmup_steps=params.get("warmup_steps", 1000),
                total_steps=params.get("total_steps", 10000),
                eta_min=params.get("eta_min", 1e-6)
            )

        # For other PyTorch schedulers
        class_path = spec.get("class", "")
        parts = class_path.split(".")
        module = torch
        for part in parts[1:-1]:
            module = getattr(module, part)
        scheduler_class = getattr(module, parts[-1])
        return scheduler_class(optimizer, **params)


def create_scheduler(scheduler_name, optimizer, **param_overrides):
    return SchedulerFactory.create(scheduler_name, optimizer, param_overrides or None)
