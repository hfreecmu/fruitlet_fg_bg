import torch.nn as nn
import torch

from models.pointnet import PointNetfeat

### mlp
def fc(in_dim, out_dims):
    layers = []
    prev_dim = in_dim
    for i in range(len(out_dims) - 1):
        fc = nn.Linear(prev_dim, out_dims[i])
        #relu = nn.ReLU()
        relu = nn.LeakyReLU(0.01)

        layers.append(fc)
        layers.append(relu)

        prev_dim = out_dims[i]

    final_fc = nn.Linear(prev_dim, out_dims[-1])
    layers.append(final_fc)

    return nn.Sequential(*layers)

class MLP(nn.Module):
    def __init__(self, in_dim, out_dims):
        super(MLP, self).__init__()

        self.network = fc(in_dim, out_dims)

    def forward(self, x):
        x = self.network(x)
        return x
###

class FgBgClassifier(nn.Module):
    def __init__(self):
        
        super(FgBgClassifier, self).__init__()
        self.pointnet_feat = PointNetfeat(global_feat=True, feature_transform=False)
        self.gnn = torch.nn.Transformer(d_model=512, num_encoder_layers=1, dim_feedforward=512, 
                                        batch_first=True).encoder
        self.mlp = MLP(512, [64, 1])

    #clouds is list of [1 x 3 x n]
    def forward(self, clouds, is_tag):
        cloud_features = []
        for cloud in clouds:
            cloud_feature, _, _ = self.pointnet_feat(cloud)
            cloud_features.append(cloud_feature)

        cloud_features = torch.concatenate(cloud_features)

        #TODO confirm this does not kill gradients
        cloud_features = torch.concatenate((cloud_features, is_tag.reshape(is_tag.shape[0], -1)), dim=-1)
        
        cloud_features = self.gnn(cloud_features)
        is_fg = self.mlp(cloud_features)
        
        return is_fg