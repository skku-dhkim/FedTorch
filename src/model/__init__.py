from .custom_cnn import CustomCNN
from .skeleton import FederatedModel


def model_call(model_name: str, num_of_classes: int) -> FederatedModel:
    if model_name.lower() == 'custom_cnn':
        return CustomCNN(num_of_classes=num_of_classes)
    else:
        raise NotImplementedError("Not implemented yet.")


__all__ = [
    'CustomCNN',
    'FederatedModel',
    'model_call'
]
