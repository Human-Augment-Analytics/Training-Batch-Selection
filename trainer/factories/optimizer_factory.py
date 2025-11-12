import torch
import torch.optim as optim
from trainer.constants_models import OPTIMIZER_SPECS, get_optimizer_spec


class OptimizerFactory:
    @staticmethod
    def create(optimizer_name, model_parameters, param_overrides=None):
        spec = get_optimizer_spec(optimizer_name)
        class_path = spec["class"]

        try:
            parts = class_path.split(".")
            module = torch
            for part in parts[1:-1]:
                module = getattr(module, part)
            optimizer_class = getattr(module, parts[-1])
        except AttributeError as e:
            raise AttributeError(f"Optimizer {class_path} not found: {e}")

        params = spec["params"].copy()
        if param_overrides:
            params.update(param_overrides)

        return optimizer_class(model_parameters, **params)

    @staticmethod
    def list_optimizers():
        return list(OPTIMIZER_SPECS.keys())


def create_optimizer(optimizer_name, model_parameters, **param_overrides):
    return OptimizerFactory.create(optimizer_name, model_parameters,
                                   param_overrides or None)
