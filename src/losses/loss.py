from torch import nn
import torch


class FeatureBalanceLoss(nn.Module):
    def __init__(self, total_local_epochs, num_per_class, curr):
        super(FeatureBalanceLoss, self).__init__()
        self.total_local_epochs = total_local_epochs
        self.num_per_class = num_per_class
        lambda_ = torch.nan_to_num(torch.log(torch.Tensor(num_per_class)), neginf=0)
        self.lambda_ = lambda_.max() - lambda_
        self.lambda_ = self.lambda_ / torch.sum(self.lambda_).unsqueeze(0)
        # self.alpha = pow((curr/total_local_epochs), 2)

    def forward(self, out, global_out, label, feature_map, curr):
        norm_vector = feature_map.view(feature_map.size()[0], -1)
        norm_vector = torch.norm(norm_vector, p=2, dim=-1).unsqueeze(1)
        alpha = pow((curr/self.total_local_epochs), 2)
        logit = out - (alpha*(self.lambda_/(norm_vector+1e-12)))

        # alpha = 0.5
        # _out = torch.clone(out).detach()
        # _out = nn.functional.softmax(_out, dim=-1)
        # _global_out = nn.functional.softmax(global_out).detach()
        # changes = nn.functional.softmax(torch.exp(-(_out - _global_out)), dim=-1)
        # print(label.size())
        # print(nilloss.size())
        # changes = torch.exp(-(_out - global_out)) / torch.sum(torch.exp(-(_out - global_out)), dim=-1)
        # alpha = pow((curr/self.total_local_epochs), 2)
        # alpha = curr/self.total_local_epochs

        # alpha = 1
        # logit = out - alpha*changes/(norm_vector+1e-12)

        # return torch.nan_to_num(nn.functional.cross_entropy(logit, label))
        return torch.nn.functional.cross_entropy(logit, label)
