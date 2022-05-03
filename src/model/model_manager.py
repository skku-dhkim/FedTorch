import torchvision.models as model
import src.model.custom_cnn as cnn
import torch


def get_model(model_name, num_of_classes):
    model_name = model_name.lower()
    if model_name == "resnet-50":
        _model = model.resnet50()
        fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=2048, out_features=1000, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=1000, out_features=num_of_classes, bias=True)
        )
        _model.fc = fc
        return _model
    elif model_name == "vgg-11":
        return model.vgg11()
    elif model_name == "custom_cnn":
        return cnn.CustomCNN(num_classes=num_of_classes)
    elif model_name == "mini_cnn":
        return cnn.MiniCustomCNN()
