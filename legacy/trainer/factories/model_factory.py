import importlib
import torch.nn as nn
from trainer.constants_models import MODEL_SPECS, get_model_spec


class ModelFactory:
    @staticmethod
    def create(model_name, param_overrides=None):
        spec = get_model_spec(model_name)
        module_name = spec["module"]
        class_name = spec["class"]

        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            raise ImportError(f"Can't import {module_name}: {e}")

        try:
            model_class = getattr(module, class_name)
        except AttributeError as e:
            raise AttributeError(f"Class {class_name} not found: {e}")

        params = spec["params"].copy()
        if param_overrides:
            params.update(param_overrides)

        return model_class(**params)

    @staticmethod
    def get_model_info(model_name):
        return get_model_spec(model_name)

    @staticmethod
    def list_models(model_type=None):
        if model_type:
            return [name for name, spec in MODEL_SPECS.items()
                   if spec.get("type") == model_type]
        return list(MODEL_SPECS.keys())


def create_model(model_name, **param_overrides):
    return ModelFactory.create(model_name, param_overrides or None)
