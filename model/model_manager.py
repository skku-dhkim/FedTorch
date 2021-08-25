import torchvision.models as model
import model.custom_cnn as cnn
import torch


def get_model(model_name):
    model_name = model_name.lower()
    if model_name == "resnet-50":
        _model = model.resnet50()
        fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=2048, out_features=1000, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=1000, out_features=10, bias=True)
        )
        _model.fc = fc
        return _model
    elif model_name == "vgg-11":
        return model.vgg11()
    elif model_name == "custom_cnn":
        return cnn.CustomCNN()
