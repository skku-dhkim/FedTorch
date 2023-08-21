from torch import nn
import torch


class FeatureBalanceLoss(nn.Module):
    def __init__(self, total_local_epochs):
        super(FeatureBalanceLoss, self).__init__()
        self.total_local_epochs = total_local_epochs

    def forward(self, out, global_out, label, feature_map, curr):
        norm_vector = feature_map.view(feature_map.size()[0], -1)
        norm_vector = torch.norm(norm_vector, p=2, dim=-1).unsqueeze(1)
        global_out = nn.functional.softmax(global_out.detach(), dim=-1)
        out = nn.functional.softmax(out, dim=-1)
        changes = nn.functional.softmax(torch.exp(-(out - global_out)), dim=-1)
        alpha = pow((curr/self.total_local_epochs), 2)
        logit = out - alpha*changes/(norm_vector+1e-12)
        return nn.functional.cross_entropy(logit, label)
