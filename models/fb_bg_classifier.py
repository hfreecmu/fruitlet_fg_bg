import torch.nn as nn
import torch
from models.encoder import DescriptorEncoder, KeypointEncoder

### mlp
def fc(in_dim, out_dims, norm):
    layers = []
    prev_dim = in_dim
    for i in range(len(out_dims) - 1):
        fc = nn.Linear(prev_dim, out_dims[i])
        relu = nn.ReLU()
        #relu = nn.LeakyReLU(0.01)

        layers.append(fc)
        if norm == "instance":
            layers.append(nn.InstanceNorm1d(out_dims[i]))
        elif norm == "batch":
            layers.append(nn.BatchNorm1d(out_dims[i]))
        elif norm is None:
            pass
        else:
            raise RuntimeError('Illagel norm passed: ' + norm)
        layers.append(relu)

        prev_dim = out_dims[i]

    final_fc = nn.Linear(prev_dim, out_dims[-1])
    layers.append(final_fc)

    return nn.Sequential(*layers)

class MLP(nn.Module):
    def __init__(self, params):
        super(MLP, self).__init__()

        input_dim = params['input_dim']
        output_dims = params['output_dims']
        norm = params['norm']

        self.network = fc(input_dim, output_dims, norm)

    def forward(self, x):
        x = self.network(x)
        return x
###

### gnn
def attention(query, key, value):
    dim = query.shape[3]
    scores = torch.einsum('bnhd,bmhd->bhnm', query, key) / dim**.5

    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bmhd->bnhd', prob, value), prob

#I disagreed with how this was originally written
#so I re-wrote it
class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads #dk and dv
        self.num_heads = num_heads

        self.merge = MLP({'input_dim': d_model, 'output_dims': [d_model], 'norm': None})
        self.projQ = nn.ModuleList([MLP({'input_dim': d_model, 'output_dims': [self.dim], 'norm': None}) for _ in range(num_heads)]) #dmodelxdk
        self.projK = nn.ModuleList([MLP({'input_dim': d_model, 'output_dims': [self.dim], 'norm': None}) for _ in range(num_heads)]) #dmodelxdk
        self.projV = nn.ModuleList([MLP({'input_dim': d_model, 'output_dims': [self.dim], 'norm': None}) for _ in range(num_heads)]) #dmodelxdv

        # self.merge = nn.Conv1d(d_model, d_model, kernel_size=1) #h*dvxdmodel
        # #self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])
        # self.projQ = nn.ModuleList([nn.Conv1d(d_model, self.dim, kernel_size=1) for _ in range(num_heads)]) #dmodelxdk
        # self.projK = nn.ModuleList([nn.Conv1d(d_model, self.dim, kernel_size=1) for _ in range(num_heads)]) #dmodelxdk
        # self.projV = nn.ModuleList([nn.Conv1d(d_model, self.dim, kernel_size=1) for _ in range(num_heads)]) #dmodelxdv

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        # query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
        #                      for l, x in zip(self.proj, (query, key, value))]
        queries = [q(query) for q in self.projQ]
        keys = [k(key)for k in self.projK]
        values = [v(value) for v in self.projV]

        query = torch.stack(queries, dim=2)
        key = torch.stack(keys, dim=2)
        value = torch.stack(values, dim=2)

        x, _ = attention(query, key, value)

        return self.merge(x.contiguous().view(batch_dim, -1, self.dim*self.num_heads))

class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int, norm: str):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP({'input_dim': feature_dim*2, 
                        'output_dims': [feature_dim*2, feature_dim],
                        'norm': norm})
        #removed
        #nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x):
        message = self.attn(x, x, x)
        return self.mlp(torch.cat([x, message], dim=2))
    
class AttentionalGNN(nn.Module):
    def __init__(self, params):
        super().__init__()

        num_layer_reps = params['num_layer_reps']
        d_model = params['d_model']
        num_heads = params['num_heads']
        norm = params['norm']
        
        self.layers = nn.ModuleList([
            AttentionalPropagation(d_model, num_heads, norm)
            for _ in range(num_layer_reps)])
        
    def forward(self, descs):
        for layer in self.layers:
            delta = layer(descs)

            descs = (descs + delta)

        return descs
###

class FgBgClassifier(nn.Module):
    def __init__(self, params):
        super(FgBgClassifier, self).__init__()
        
        gnn_params = params['gnn_params']

        final_mlp_params = {
            'input_dim': gnn_params['d_model'],
            'output_dims': params['final_mlp'],
            'norm': None
        }
        
        self.denc = DescriptorEncoder()
        self.kenc = KeypointEncoder()

        self.gnn = AttentionalGNN(gnn_params)
        self.final_mlp = MLP(final_mlp_params)

        self.feature_scale = gnn_params['d_model']**0.25

    def forward(self, descs, kpts, is_tags, scores):

        descs = self.denc(descs).squeeze(-1).squeeze(-1)
        kpts = self.kenc(kpts).squeeze(-1).squeeze(-1)

        descs = descs + kpts
        descs = torch.concatenate([descs, is_tags.unsqueeze(-1), scores.unsqueeze(-1)], dim=1)

        descs = self.gnn(descs.unsqueeze(0))

        descs = self.final_mlp(descs)
        #descs = descs / self.feature_scale
        
        return descs.squeeze(0)