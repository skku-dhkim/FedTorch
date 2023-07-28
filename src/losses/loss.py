from torch import nn
import torch


class FeatureBalanceLoss(nn.Module):
    def __init__(self, class_num_list: list, total_global_epochs, curr):
        super(FeatureBalanceLoss, self).__init__()
        class_num_tensor = torch.FloatTensor(class_num_list)
        intensity_lambda = torch.nan_to_num(torch.log(class_num_tensor), posinf=0, neginf=0)
        self.intensity_lambda = intensity_lambda.max() - intensity_lambda
        self.total_global_epochs = total_global_epochs
        # NOTE: Minus one from current global iteration because it strats from 1 instead of 0.
        curr -= 1
        self.alpha = pow((curr/self.total_global_epochs), 2)

    def forward(self, out, label, feature_map):
        norm_vector = feature_map.view(feature_map.size()[0], -1)
        norm_vector = torch.norm(norm_vector, p=2, dim=-1).unsqueeze(1)
        _lambda = torch.tile(self.intensity_lambda.unsqueeze(0), (out.size()[0], 1))
        logit = out - self.alpha*_lambda/(norm_vector+1e-12)

        return (1-self.alpha)*nn.functional.cross_entropy(logit, label)
