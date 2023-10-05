import torch.nn as nn
import torch

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
        self.cloud_mlp = MLP(4, [64, 256, 512])
        self.cloud_gnn = torch.nn.Transformer(d_model=512, num_encoder_layers=3, dim_feedforward=512, 
                                              batch_first=True).encoder
        self.pos_enc_mlp = MLP(4, [64, 256, 512])
        
        self.gnn = torch.nn.Transformer(d_model=512, num_encoder_layers=2, dim_feedforward=512, 
                                        batch_first=True).encoder
        self.mlp = MLP(512, [64, 8, 1])

    #clouds is list of [1 x 3 x n]
    def forward(self, clouds, is_tag):
        #step 1) get centroid of all cloud for offset
        full_centroid = torch.concatenate(clouds, dim=-1).mean(dim=(0,2))

        #step 2) transform each cloud
        cloud_features = []
        cloud_centroids = []
        for i in range(len(clouds)):
            cloud = clouds[i]
            cloud_centroid = cloud.mean(dim=(0,2))
            centered_cloud = cloud - cloud_centroid.reshape((1, 3, -1))
            tag_feature = torch.ones((1, 1, centered_cloud.shape[-1])).float().to(cloud_centroid.device)*is_tag[i]
            cloud_feature = torch.concatenate((centered_cloud, tag_feature), dim=1)
            cloud_feature = torch.permute(cloud_feature, (0, 2, 1))
            cloud_feature = self.cloud_mlp(cloud_feature.squeeze(0)).unsqueeze(0)
            cloud_feature = self.cloud_gnn(cloud_feature)
            cloud_feature = torch.max(cloud_feature, 1, keepdim=True)[0][:, 0, :]

            cloud_features.append(cloud_feature)
            cloud_centroids.append(cloud_centroid - full_centroid)
        
        #step 3) get positional encoding
        cloud_centroids = torch.stack(cloud_centroids)
        cloud_centroids = torch.concatenate((cloud_centroids, is_tag.reshape((-1, 1))), dim=-1)
        pos_enc = self.pos_enc_mlp(cloud_centroids)

        #step 4) add positional encoding and cloud features
        cloud_features = torch.concatenate(cloud_features)

        x = cloud_features + pos_enc
        x = x.unsqueeze(0)
        x = self.gnn(x)
        is_fg = self.mlp(x.squeeze(0))
        
        return is_fg