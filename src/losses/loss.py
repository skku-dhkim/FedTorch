from torch import nn
import torch


class FeatureBalanceLoss(nn.Module):
    def __init__(self, total_local_epochs, num_per_class):
        super(FeatureBalanceLoss, self).__init__()
        self.total_local_epochs = total_local_epochs
        self.num_per_class = num_per_class
        lambda_ = torch.nan_to_num(torch.log(torch.Tensor(num_per_class)), neginf=0)
        self.lambda_ = lambda_.max() - lambda_
        self.lambda_ = self.lambda_ / torch.sum(self.lambda_).unsqueeze(0)

    def forward(self, out, label, feature_map, curr, device=None):
        if device is None:
            device = "cpu"
        norm_vector = feature_map.view(feature_map.size()[0], -1)
        norm_vector = torch.norm(norm_vector, p=2, dim=-1).unsqueeze(1).to(device)
        alpha = pow((curr/self.total_local_epochs), 2)
        logit = out - (alpha*(self.lambda_.to(device)/(norm_vector+1e-12)))
        return torch.nn.functional.cross_entropy(logit, label)
