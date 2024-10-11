import warnings
warnings.filterwarnings('ignore')
import os
import pdb
import copy
import os.path as osp
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
import numpy        as np
import numpy.random as npr
import torch
from torch import nn
from torch.optim import Adam
from GCL.eval import get_split#, LREvaluator
from abc import ABC, abstractmethod
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.svm import LinearSVC, SVC
import GCL.losses as L
import GCL.augmentors as A
from GCL.augmentors.augmentor import Graph, Augmentor
from GCL.augmentors.functional import add_edge
import time
import torch_geometric.transforms as T
import pickle as pk
import uuid
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    SparseTensor,
)
from scipy.sparse import coo_matrix
from tqdm import tqdm
from GCL.models import DualBranchContrast
from torch_geometric.nn import GCNConv, global_add_pool, GINConv, GATv2Conv, GPSConv,MLP
from torch_geometric.nn.inits import uniform
from torch_geometric.datasets import Planetoid,TUDataset

from torch_geometric.utils import dropout_adj, to_networkx, to_undirected, degree, to_scipy_sparse_matrix, \
    from_scipy_sparse_matrix, sort_edge_index, add_self_loops, subgraph, get_laplacian, scatter,is_torch_sparse_tensor,dense_to_sparse, to_dense_adj
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.transforms import ToSparseTensor
from GCL.models.contrast_model import WithinEmbedContrast
# from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR

import random
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)

'''
def maybe_num_nodes(index, num_nodes=None):
    return index.max().item() + 1 if num_nodes is None else num_nodes
'''

def get_graph_store_path(args):
    save_path = os.path.join('graph_ckpt', args.method)
    os.makedirs(save_path, exist_ok=True)

    res_path = os.path.join(
        save_path,
        f'{args.method}_{args.epochs}epochs_random_search_{args.num_layers}layers'
    )

    return res_path


def getDegree(edge_index, edge_weight =None, num_nodes=None):
    
    '''
    Require: edge_index of the dataset(2 * num_edges)
                    (optional: edge weight of edges , tensor(num_edges))
                    
    Return: the degree of the nodes: tensor(num_edges)
    '''
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1),
                             device=edge_index.device)

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, row, 0, dim_size=num_nodes, reduce='sum')
    
    return deg


## convert edge_index into adjacency matrix
def convert(dim, edge_index, edge_weight=None):
    '''
    Require:  dim: the number of node of the graph
                    edge_index of the dataset(2 * num_edges)
                    (optional: edge weight of edges , tensor(num_edges))
                    
    Return: The adj mat of the graph: tensor(dim * dim)
    '''
    adj_mat = torch.zeros([dim,dim])
    size = edge_index.shape[1]
    if edge_weight is None:
        for i in range(size):
            adj_mat[edge_index[0][i],edge_index[1][i]] = 1
    else:
        for i in range(size):
            adj_mat[edge_index[0][i],edge_index[1][i]] = edge_weight[i]
    
    return adj_mat

def calculate_spectrum(edge_index, edge_weight=None, num_nodes=None, smallest_eigvals=None):

    # Compute the Laplacian matrix
    laplacian_index, laplacian_weight  = get_laplacian(edge_index, edge_weight, num_nodes=num_nodes, normalization='sym')

    # Convert sparse Laplacian matrix to dense format
    laplacian_dense = to_dense_adj(edge_index=laplacian_index, edge_attr=laplacian_weight, max_num_nodes=num_nodes).squeeze()

    # Calculate eigenvalues using torch.linalg.eigh, which returns eigenvalues and eigenvectors
    try:
        eigenvalues, _ = torch.linalg.eigh(laplacian_dense)
    except Exception as e:
        print(e)
        eigenvalues = torch.tensor([])

    # Since eigenvalues returned by eigh are in ascending order, we take the smallest ones
    if smallest_eigvals is None:
        return eigenvalues
    return eigenvalues[:smallest_eigvals]

class FromGraphsAugmentorNormal(Augmentor):
    def __init__(self, graphs, perturb_ratio=-1, min_perturb_l2=0.5):
        super(FromGraphsAugmentorNormal, self).__init__()
        self.graphs = graphs
        self.curr_graph_index = 0
        self.perturb_ratio = perturb_ratio
        self.min_perturb_l2 = min_perturb_l2
        self.perturb_aug = A.EdgeRemoving(pe=perturb_ratio)
        self.l2s = []

    def augment(self, g: Graph, batch: torch.Tensor=None) -> Graph:
        curr_graph = self.graphs[self.curr_graph_index]
        self.curr_graph_index = (self.curr_graph_index + 1) % len(self.graphs)

        if self.perturb_ratio > 0:
            curr_l2 = 0
            curr_best = None
            original_spec = calculate_spectrum(
                edge_index=curr_graph.edge_index,
                edge_weight=curr_graph.edge_weights,
            )
            for i in range(200):
                new_x, new_edge_index, new_edge_weight = self.perturb_aug(
                    curr_graph.x,
                    curr_graph.edge_index,
                    curr_graph.edge_weights,
                )
                new_spec = calculate_spectrum(
                    edge_index=new_edge_index,
                    edge_weight=new_edge_weight,
                )
                new_l2 = np.linalg.norm(original_spec.cpu() - new_spec.cpu(), 2)
                if new_l2 > curr_l2:
                    curr_l2 = new_l2
                    curr_best = Graph(
                        x=new_x, 
                        edge_index=new_edge_index, 
                        edge_weights=new_edge_weight
                    )
                
                if curr_l2 > self.min_perturb_l2:
                    print(curr_l2)
                    break
            if curr_l2 <= self.min_perturb_l2:
                print(f'Failed to perturb the graph to the desired L2 distance. Best so far: {curr_l2}')
                
            self.l2s.append(curr_l2)
            return curr_best

        return curr_graph    

def log_data(encoder_model,ori, model_name, data_name,norm, rate, epsilon, test_result,epoch, aug1=None, aug2=None):
    
    path = f'{model_name}-{ori}-{data_name}-isNorm{norm}-rate={str(rate)}-epsilon={epsilon}-F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}-epoch{str(epoch)}'
    os.mkdir(path)
    
    
    f = open(path+'/encoder.pkl', 'wb')
    pk.dump(encoder_model,f)
    f.close()
    
    if aug1 is not None:
        f = open(path+'/aug1.pkl', 'wb')
        pk.dump(aug1,f)
        f.close() 
    if aug2 is not None:        
        f = open(path+'/aug2.pkl', 'wb')
        pk.dump(aug2,f)
        f.close() 
        
    return path
        
def log_config(path,op_list):
    for i, op in enumerate(op_list):
        f = open(path+'/op-'+str(i)+'-.pkl', 'wb')
        pk.dump(op,f)
        f.close()
        

class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict) -> dict:
        pass

    def __call__(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict) -> dict:
        for key in ['train', 'test', 'valid']:
            assert key in split

        result = self.evaluate(x, y, split)
        return result

class LogisticRegression(nn.Module):
    def __init__(self, num_features, num_classes):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(num_features, num_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x):
        z = self.fc(x)
        return z


class LREvaluator(BaseEvaluator):
    def __init__(self, num_epochs: int = 5000, learning_rate: float = 0.001,
                 weight_decay: float = 0.0, test_interval: int = 20):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.test_interval = test_interval

    def evaluate(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict):
        device = x.device
        x = x.detach().to(device)
        input_dim = x.size()[1]
        y = y.to(device)
        num_classes = y.max().item() + 1
        classifier = LogisticRegression(input_dim, num_classes).to(device)
        optimizer = Adam(classifier.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        output_fn = nn.LogSoftmax(dim=-1)
        criterion = nn.NLLLoss()
        
        best_val_acc = 0
        best_val_micro = 0
        best_val_macro = 0
        best_test_acc = 0
        best_test_micro = 0
        best_test_macro = 0
        best_epoch = 0

        with tqdm(total=self.num_epochs, desc='(LR)+{:.4f}decay'.format(self.weight_decay),
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]') as pbar:
            for epoch in range(self.num_epochs):
                classifier.train()
                optimizer.zero_grad()

                output = classifier(x[split['train']])
                loss = criterion(output_fn(output), y[split['train']])

                loss.backward()
                optimizer.step()

                if (epoch + 1) % self.test_interval == 0:
                    classifier.eval()
                    y_test = y[split['test']].detach().cpu().numpy()
                    y_pred = classifier(x[split['test']]).argmax(-1).detach().cpu().numpy()
                    test_acc = accuracy_score(y_test, y_pred)
                    test_micro = f1_score(y_test, y_pred, average='micro')
                    test_macro = f1_score(y_test, y_pred, average='macro')

                    y_val = y[split['valid']].detach().cpu().numpy()
                    y_pred = classifier(x[split['valid']]).argmax(-1).detach().cpu().numpy()
                    val_acc = accuracy_score(y_val, y_pred)
                    val_micro = f1_score(y_val, y_pred, average='micro')
                    val_macro = f1_score(y_val, y_pred, average='macro')

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_val_micro = val_micro
                        best_val_macro = val_macro
                        best_test_acc = test_acc
                        best_test_micro = test_micro
                        best_test_macro = test_macro
                        best_epoch = epoch

                    pbar.set_postfix({'Test accuracy': best_test_acc, 
                                      'F1Mi': best_test_micro, 
                                      'F1Ma': best_test_macro})
                    pbar.update(self.test_interval)

        return {
            'accuracy': best_test_acc,
            'micro_f1': best_test_micro,
            'macro_f1': best_test_macro,
            'accuracy_val': best_val_acc,
            'micro_f1_val': best_val_micro,
            'macro_f1_val': best_val_macro
        }



class GraceGConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers,GNN_model='GCN'):
        super(GraceGConv, self).__init__()
        self.GNN_model=GNN_model
        self.layers = torch.nn.ModuleList()
        self.activation = nn.ReLU()
        for i in range(num_layers):
            if i == 0:
                if GNN_model=='GCN':
                    self.layers.append(GCNConv(input_dim, hidden_dim))
                elif GNN_model=='GAT':
                    self.layers.append(GATv2Conv(input_dim, hidden_dim,heads=2,concat=False))
                elif GNN_model=='GPS':
                    self.mlp_0 = MLP([input_dim, hidden_dim, hidden_dim])
                    mlp = MLP([hidden_dim, hidden_dim, hidden_dim])
                    self.layers.append(GPSConv(hidden_dim, GINConv(mlp), heads=2))                
            else:
                if GNN_model=='GCN':
                    self.layers.append(GCNConv(hidden_dim, hidden_dim))
                elif GNN_model=='GAT':
                    self.layers.append(GATv2Conv(hidden_dim, hidden_dim,heads=2,concat=False))
                elif GNN_model=='GPS':
                    mlp = MLP([hidden_dim, hidden_dim, hidden_dim])
                    self.layers.append(GPSConv(hidden_dim, GINConv(mlp), heads=2))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        if  self.GNN_model=='GPS':
            z=self.mlp_0(x)
        for conv in self.layers:
            if self.GNN_model=='GCN':
                z = conv(z, edge_index,edge_weight=edge_weight)
                z = self.activation(z)
            else:
                z = conv(z, edge_index)   
                z = self.activation(z)
        return z

class GraceEncoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, proj_dim):
        super(GraceEncoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        z = self.encoder(x, edge_index, edge_weight)
        z1 = self.encoder(x1, edge_index1, edge_weight1)
        z2 = self.encoder(x2, edge_index2, edge_weight2)
        return z, z1, z2

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)


class GBT_GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers,GNN_model):
        super(GBT_GConv, self).__init__()
        self.act = torch.nn.PReLU()
        self.bn = torch.nn.BatchNorm1d(2 * hidden_dim, momentum=0.01)
        self.GNN_model=GNN_model
        if GNN_model=='GCN':
            self.conv1 = GCNConv(input_dim, 2 * hidden_dim, cached=False)
            self.conv2 = GCNConv(2 * hidden_dim, hidden_dim, cached=False)
        if GNN_model=='GAT':
            self.conv1=GATv2Conv(input_dim, 2 * hidden_dim,heads=2,concat=False)
            self.conv2=GATv2Conv(2 * hidden_dim, hidden_dim,heads=2,concat=False)
        if GNN_model=='GPS':
                    self.mlp_0 = MLP([input_dim, hidden_dim, hidden_dim])
                    mlp_1 = MLP([hidden_dim, hidden_dim, hidden_dim])
                    self.conv1=GPSConv(hidden_dim, GINConv(mlp_1), heads=2)                   
                    mlp_2 = MLP([hidden_dim, hidden_dim, hidden_dim])
                    self.conv2=GPSConv(hidden_dim, GINConv(mlp_2), heads=2)
        self.layers = torch.nn.ModuleList()


        self.layers.append(self.conv1)
        for _ in range(num_layers - 2):
            if GNN_model=='GCN':
                self.layers.append(GCNConv(2 * hidden_dim, 2 * hidden_dim, cached=False))
            if GNN_model=='GAT':
                self.layers.append(GATv2Conv(2 * hidden_dim, 2 * hidden_dim,heads=2,concat=False))
            if GNN_model=='GPS':
                mlp = MLP([hidden_dim, hidden_dim, hidden_dim])
                self.layers.append(GPSConv(hidden_dim, GINConv(mlp), heads=2))
        self.layers.append(self.conv2)



    def forward(self, x, edge_index, edge_weight=None):
        if self.GNN_model=='GPS':
            z=self.mlp_0(z)

        if self.GNN_model=='GCN':
            z = self.layers[0](x, edge_index, edge_weight)
        elif self.GNN_model=='GAT':
            z = self.layers[0](x, edge_index)
        z = self.bn(z)
        z = self.act(z)
        for conv in self.layers[1:-1]:
            if self.GNN_model=='GCN':
                z = conv(z, edge_index, edge_weight)
            else:
                z = conv(z, edge_index)
            z = self.bn(z)
            z = self.act(z)
        if self.GNN_model=='GCN':
            z = self.layers[-1](z, edge_index, edge_weight)
        else:
            z = self.layers[-1](z, edge_index)
        return z


class MVGRLEncoder(torch.nn.Module):
    def __init__(self, encoder1, encoder2, augmentor, hidden_dim):
        super(MVGRLEncoder, self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.augmentor = augmentor
        self.project = torch.nn.Linear(hidden_dim, hidden_dim)
        uniform(hidden_dim, self.project.weight)

    @staticmethod
    def corruption(x, edge_index, edge_weight):
        return x[torch.randperm(x.size(0))], edge_index, edge_weight

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        z1 = self.encoder1(x1, edge_index1, edge_weight1)
        z2 = self.encoder2(x2, edge_index2, edge_weight2)
        g1 = self.project(torch.sigmoid(z1.mean(dim=0, keepdim=True)))
        g2 = self.project(torch.sigmoid(z2.mean(dim=0, keepdim=True)))
        z1n = self.encoder1(*self.corruption(x1, edge_index1, edge_weight1))
        z2n = self.encoder2(*self.corruption(x2, edge_index2, edge_weight2))
        return z1, z2, g1, g2, z1n, z2n


## the encoder
class GBT_Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, log_spectrum=False):
        super(GBT_Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor
        self.log_spectrum = log_spectrum
        if self.log_spectrum:
            self.spectrums = []

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        if self.log_spectrum:
            with torch.no_grad():
                s1 = calculate_spectrum(edge_index1, edge_weight=edge_weight1, num_nodes=x1.shape[0], smallest_eigvals=None)
                s2 = calculate_spectrum(edge_index2, edge_weight=edge_weight2, num_nodes=x2.shape[0], smallest_eigvals=None)
                self.spectrums.append((s1, s2))

        z = self.encoder(x, edge_index, edge_weight)
        z1 = self.encoder(x1, edge_index1, edge_weight1)
        z2 = self.encoder(x2, edge_index2, edge_weight2)
        return z, z1, z2


from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import os.path as osp
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Coauthor, Amazon
def get_dataset(path, name):
    assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP', 'Karate', 'WikiCS', 'Coauthor-CS', 'Coauthor-Phy',
                    'Computers', 'Photo', 'ogbn-arxiv', 'ogbg-code','MUTAG',
                     'PROTEINS',
                     'IMDB-BINARY',
                     'IMDB-MULTI',
                     'REDDIT-BINARY',
                     'NCI1',
                     'DD',
                     'COLLAB',
                     'PTC_MR']
    name = 'dblp' if name == 'DBLP' else name

    if name == 'Coauthor-CS':
        return Coauthor(root=path, name='cs', transform=T.NormalizeFeatures())

    if name == 'Coauthor-Phy':
        return Coauthor(root=path, name='physics', transform=T.NormalizeFeatures())

    if name == 'WikiCS':
        return WikiCS(root=path, transform=T.NormalizeFeatures())

    if name == 'Computers':
        return Amazon(root=path, name='Computers', transform=T.NormalizeFeatures())

    if name == 'Photo':
        return Amazon(root=path, name='Photo', transform=T.NormalizeFeatures())

    if name.startswith('ogbn'):
        return PygNodePropPredDataset(root=osp.join(path, 'OGB'), name=name, transform=T.NormalizeFeatures())

    if name == 'dblp':
         return CitationFull(osp.join(path, 'Citation'), name, transform=T.NormalizeFeatures())

    if name in ['Cora', 'CiteSeer', 'PubMed']:
        return Planetoid(path, name, transform=T.NormalizeFeatures())
    if name in ['MUTAG','PROTEINS','IMDB-BINARY','IMDB-MULTI','REDDIT-BINARY','NCI1','DD','COLLAB','PTC_MR']:
        return TUDataset(path, name, transform=T.NormalizeFeatures())



def make_gin_conv(input_dim: int, out_dim: int) -> GINConv:
    mlp = torch.nn.Sequential(
        torch.nn.Linear(input_dim, out_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(out_dim, out_dim))
    return GINConv(mlp)

class Normalize(torch.nn.Module):
    def __init__(self, dim=None, norm='batch'):
        super().__init__()
        if dim is None or norm == 'none':
            self.norm = lambda x: x
        if norm == 'batch':
            self.norm = torch.nn.BatchNorm1d(dim)
        elif norm == 'layer':
            self.norm = torch.nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)

class G2LBGRLEncoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, dropout=0.2, predictor_norm='batch', log_spectrum = False):
        super(G2LBGRLEncoder, self).__init__()
        self.online_encoder = encoder
        self.target_encoder = None
        self.augmentor = augmentor
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            Normalize(hidden_dim, norm=predictor_norm),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))
        self.log_spectrum = log_spectrum
        if self.log_spectrum:
            self.spectrums = []

    def get_target_encoder(self):
        if self.target_encoder is None:
            self.target_encoder = copy.deepcopy(self.online_encoder)

            for p in self.target_encoder.parameters():
                p.requires_grad = False
        return self.target_encoder

    def update_target_encoder(self, momentum: float):
        for p, new_p in zip(self.get_target_encoder().parameters(), self.online_encoder.parameters()):
            next_p = momentum * p.data + (1 - momentum) * new_p.data
            p.data = next_p

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)

        if self.log_spectrum:
            with torch.no_grad():
                s1 = calculate_spectrum(edge_index1, edge_weight=edge_weight1, num_nodes=x1.shape[0], smallest_eigvals=None)
                s2 = calculate_spectrum(edge_index2, edge_weight=edge_weight2, num_nodes=x2.shape[0], smallest_eigvals=None)
                self.spectrums.append((s1, s2))

        h1, h1_online = self.online_encoder(x1, edge_index1, edge_weight1)
        h2, h2_online = self.online_encoder(x2, edge_index2, edge_weight2)

        g1 = global_add_pool(h1, batch)
        h1_pred = self.predictor(h1_online)
        g2 = global_add_pool(h2, batch)
        h2_pred = self.predictor(h2_online)

        with torch.no_grad():
            _, h1_target = self.get_target_encoder()(x1, edge_index1, edge_weight1)
            _, h2_target = self.get_target_encoder()(x2, edge_index2, edge_weight2)
            g1_target = global_add_pool(h1_target, batch)
            g2_target = global_add_pool(h2_target, batch)

        return g1, g2, h1_pred, h2_pred, g1_target, g2_target


class L2LBGRLEncoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, dropout=0.2, predictor_norm='batch', log_spectrum = False):
        super(L2LBGRLEncoder, self).__init__()
        self.online_encoder = encoder
        self.target_encoder = None
        self.augmentor = augmentor
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            Normalize(hidden_dim, norm=predictor_norm),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))
        
        self.log_spectrum = log_spectrum
        if self.log_spectrum:
            self.spectrums = []

    def get_target_encoder(self):
        if self.target_encoder is None:
            self.target_encoder = copy.deepcopy(self.online_encoder)

            for p in self.target_encoder.parameters():
                p.requires_grad = False
        return self.target_encoder

    def update_target_encoder(self, momentum: float):
        for p, new_p in zip(self.get_target_encoder().parameters(), self.online_encoder.parameters()):
            next_p = momentum * p.data + (1 - momentum) * new_p.data
            p.data = next_p

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)

        if self.log_spectrum:
            with torch.no_grad():
                s1 = calculate_spectrum(edge_index1, edge_weight=edge_weight1, num_nodes=x1.shape[0], smallest_eigvals=None)
                s2 = calculate_spectrum(edge_index2, edge_weight=edge_weight2, num_nodes=x2.shape[0], smallest_eigvals=None)
                self.spectrums.append((s1, s2))

        h1, h1_online = self.online_encoder(x1, edge_index1, edge_weight1)
        h2, h2_online = self.online_encoder(x2, edge_index2, edge_weight2)

        h1_pred = self.predictor(h1_online)
        h2_pred = self.predictor(h2_online)

        with torch.no_grad():
            _, h1_target = self.get_target_encoder()(x1, edge_index1, edge_weight1)
            _, h2_target = self.get_target_encoder()(x2, edge_index2, edge_weight2)

        return h1, h2, h1_pred, h2_pred, h1_target, h2_target


class MVGRLGConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers,GNN_model):
        super(MVGRLGConv, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.activation = nn.PReLU(hidden_dim)
        self.GNN_model=GNN_model
        for i in range(num_layers):
            if i == 0:
                if GNN_model=='GCN':
                    self.layers.append(GCNConv(input_dim, hidden_dim))
                elif GNN_model=='GAT':
                    self.layers.append(GATv2Conv(input_dim, hidden_dim,heads=2,concat=False))
                elif GNN_model=='GPS':
                    self.mlp_0 = MLP([input_dim, hidden_dim, hidden_dim])
                    mlp = MLP([hidden_dim, hidden_dim, hidden_dim])
                    self.layers.append(GPSConv(hidden_dim, GINConv(mlp), heads=2))                
            else:
                if GNN_model=='GCN':
                    self.layers.append(GCNConv(hidden_dim, hidden_dim))
                elif GNN_model=='GAT':
                    self.layers.append(GATv2Conv(hidden_dim, hidden_dim,heads=2,concat=False))
                elif GNN_model=='GPS':
                    mlp = MLP([hidden_dim, hidden_dim, hidden_dim])
                    self.layers.append(GPSConv(hidden_dim, GINConv(mlp), heads=2))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        if  self.GNN_model=='GPS':
            z=self.mlp_0(x)
        for conv in self.layers:
            if self.GNN_model=='GCN':
                z = conv(z, edge_index,edge_weight=edge_weight)
                z = self.activation(z)
            else:
                z = conv(z, edge_index)   
                z = self.activation(z)
        return z


class G2LBGRLGConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2,
                 encoder_norm='batch', projector_norm='batch'):
        super(G2LBGRLGConv, self).__init__()
        self.activation = torch.nn.PReLU()
        self.dropout = dropout

        self.layers = torch.nn.ModuleList()
        self.layers.append(make_gin_conv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(make_gin_conv(hidden_dim, hidden_dim))

        self.batch_norm = Normalize(hidden_dim, norm=encoder_norm)
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            Normalize(hidden_dim, norm=projector_norm),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for conv in self.layers:
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
            z = F.dropout(z, p=self.dropout, training=self.training)
        z = self.batch_norm(z)
        return z, self.projection_head(z)


class L2LBGRLGConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2,
                 encoder_norm='batch', projector_norm='batch',GNN_model='GAT'):
        super(L2LBGRLGConv, self).__init__()
        self.activation = torch.nn.PReLU()
        self.dropout = dropout
        self.GNN_model=GNN_model
        self.layers = torch.nn.ModuleList()
        if GNN_model=='GCN':
                    self.layers.append(GCNConv(input_dim, hidden_dim))
        elif GNN_model=='GAT':
                    self.layers.append(GATv2Conv(input_dim, hidden_dim,heads=2,concat=False))
        elif GNN_model=='GPS':
                    self.mlp_0 = MLP([input_dim, hidden_dim, hidden_dim])
                    mlp = MLP([hidden_dim, hidden_dim, hidden_dim])
                    self.layers.append(GPSConv(hidden_dim, GINConv(mlp), heads=2))              
        for _ in range(num_layers - 1):
                if GNN_model=='GCN':
                    self.layers.append(GCNConv(hidden_dim, hidden_dim))
                elif GNN_model=='GAT':
                    self.layers.append(GATv2Conv(hidden_dim, hidden_dim,heads=2,concat=False))
                elif GNN_model=='GPS':
                    mlp = MLP([hidden_dim, hidden_dim, hidden_dim])
                    self.layers.append(GPSConv(hidden_dim, GINConv(mlp), heads=2))
        self.batch_norm = Normalize(hidden_dim, norm=encoder_norm)
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            Normalize(hidden_dim, norm=projector_norm),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        if  self.GNN_model=='GPS':
            z=self.mlp_0(x)
        for conv in self.layers:
            if self.GNN_model=='GCN':
                z = conv(z, edge_index,edge_weight=edge_weight)
                z = self.activation(z)
            else:
                z = conv(z, edge_index)   
                z = self.activation(z)
            z = F.dropout(z, p=self.dropout, training=self.training)
        z = self.batch_norm(z)
        return z, self.projection_head(z)
    
####graph level encoder
target_types = {'Edge_Adding_pergraph'}

class G_GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers,GNN_model='GPS'):
        super(G_GConv, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = nn.PReLU(hidden_dim)
        self.GNN_model=GNN_model
        for i in range(num_layers):
            if i == 0:
                if GNN_model=='GCN':
                    self.layers.append(GCNConv(input_dim, hidden_dim))
                elif GNN_model=='GAT':
                    self.layers.append(GATv2Conv(input_dim, hidden_dim,heads=2,concat=False))
                elif GNN_model=='GPS':
                    self.mlp_0 = MLP([input_dim, hidden_dim, hidden_dim])
                    mlp = MLP([hidden_dim, hidden_dim, hidden_dim])
                    self.layers.append(GPSConv(hidden_dim, GINConv(mlp), heads=2))
            else:
                if GNN_model=='GCN':
                    self.layers.append(GCNConv(hidden_dim, hidden_dim))
                elif GNN_model=='GAT':
                    self.layers.append(GATv2Conv(hidden_dim, hidden_dim,heads=2,concat=False))
                elif GNN_model=='GPS':
                    mlp = MLP([hidden_dim, hidden_dim, hidden_dim])
                    self.layers.append(GPSConv(hidden_dim, GINConv(mlp), heads=2))
                    
    def forward(self, x, edge_index, edge_weight, batch):
        z = x
        if  self.GNN_model=='GPS':
            z=self.mlp_0(x)
        zs = []
        for conv in self.layers:
            if self.GNN_model=='GCN':
                z = conv(z, edge_index,edge_weight=edge_weight)
            else:
                z = conv(z, edge_index)                
            z = self.activation(z)
            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        g = torch.cat(gs, dim=1)
        return g

class  GBTG_Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, save_dir=None):
        super(GBTG_Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor
        self.save_dir = save_dir
        self.prefix=0


    def forward(self, x, edge_index,batch,edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index,edge_weight)

        if  type(aug2).__name__ in target_types:
            x2, edge_index2, edge_weight2 = aug2(x, edge_index,batch=batch)
        else:
            x2, edge_index2, edge_weight2 = aug2(x, edge_index)        

        if self.save_dir is not None:
            prefix = str(self.prefix)
            self.prefix += 1
            graph1 = Graph(x=x1, edge_index=edge_index1, edge_weights=edge_weight1)
            graph2 = Graph(x=x2, edge_index=edge_index2, edge_weights=edge_weight2)
            pk.dump(graph1, open(os.path.join(self.save_dir, f'{prefix}_aug1.pkl'), 'wb'))
            pk.dump(graph2, open(os.path.join(self.save_dir, f'{prefix}_aug2.pkl'), 'wb'))

        z1 = self.encoder(x1, edge_index1, edge_weight1, batch=batch)
        z2 = self.encoder(x2, edge_index2, edge_weight2,batch=batch)
        return   z1, z2

###mvgrl-g###
    
class MVGRL_GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers,GNN_model='GPS'):
        super(MVGRL_GConv, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = nn.PReLU(hidden_dim)
        self.GNN_model=GNN_model
        for i in range(num_layers):
            if i == 0:
                if GNN_model=='GCN':
                    self.layers.append(GCNConv(input_dim, hidden_dim))
                elif GNN_model=='GAT':
                    self.layers.append(GATv2Conv(input_dim, hidden_dim,heads=2,concat=False))
                elif GNN_model=='GPS':
                    self.mlp_0 = MLP([input_dim, hidden_dim, hidden_dim])
                    mlp = MLP([hidden_dim, hidden_dim, hidden_dim])
                    self.layers.append(GPSConv(hidden_dim, GINConv(mlp), heads=2))
            else:
                if GNN_model=='GCN':
                    self.layers.append(GCNConv(hidden_dim, hidden_dim))
                elif GNN_model=='GAT':
                    self.layers.append(GATv2Conv(hidden_dim, hidden_dim,heads=2,concat=False))
                elif GNN_model=='GPS':
                    mlp = MLP([hidden_dim, hidden_dim, hidden_dim])
                    self.layers.append(GPSConv(hidden_dim, GINConv(mlp), heads=2))

    def forward(self, x, edge_index, edge_weight, batch):
        z = x
        if  self.GNN_model=='GPS':
            z=self.mlp_0(x)
        zs = []
        for conv in self.layers:
            if self.GNN_model=='GCN':
                z = conv(z, edge_index,edge_weight=edge_weight)
            else:
                z = conv(z, edge_index)                
            z = self.activation(z)
            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        g = torch.cat(gs, dim=1)
        return z, g


class FC(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FC, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU()
        )
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x) + self.linear(x)
    
class MVGRL_Encoder(torch.nn.Module):
    def __init__(self, gcn1, gcn2, mlp1, mlp2, aug1, aug2, save_dir=None):
        super(MVGRL_Encoder, self).__init__()
        self.gcn1 = gcn1
        self.gcn2 = gcn2
        self.mlp1 = mlp1
        self.mlp2 = mlp2
        self.aug1 = aug1
        self.aug2 = aug2
        if type(self.aug1) == type(aug2):
            self.flag=1
        else: self.flag=0
        self.save_dir = save_dir
        self.prefix = 0


    def forward(self, x, edge_index, batch):
        if type(self.aug1).__name__ in target_types:
            x1, edge_index1, edge_weight1 = self.aug1(x, edge_index,batch=batch)
        else:
            x1, edge_index1, edge_weight1 = self.aug1(x, edge_index)
        
        if type(self.aug2).__name__ in target_types:
            x2, edge_index2, edge_weight2 = self.aug2(x, edge_index,batch=batch)
        else:
            x2, edge_index2, edge_weight2 = self.aug2(x, edge_index)

        if self.save_dir is not None:
            prefix = str(self.prefix)
            self.prefix += 1
            graph1 = Graph(x=x1, edge_index=edge_index1, edge_weights=edge_weight1)
            graph2 = Graph(x=x2, edge_index=edge_index2, edge_weights=edge_weight2)
            pk.dump(graph1, open(os.path.join(self.save_dir, f'{prefix}_aug1.pkl'), 'wb'))
            pk.dump(graph2, open(os.path.join(self.save_dir, f'{prefix}_aug2.pkl'), 'wb'))

        z1, g1 = self.gcn1(x1, edge_index1, edge_weight=edge_weight1,batch=batch)
        z2, g2 = self.gcn2(x2, edge_index2, edge_weight=edge_weight2,batch=batch)
        h1, h2 = [self.mlp1(h) for h in [z1, z2]]
        g1, g2 = [self.mlp2(g) for g in [g1, g2]]
        return h1, h2, g1, g2
    
import GCL.augmentors as A
from wei_utils import GIP
class Grace_Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, proj_dim, save_dir=None, num_layers=2):
        super(Grace_Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor
        self.train_augmentor = augmentor
        self.save_dir = save_dir
        self.prefix = 0

        self.fc1 = torch.nn.Linear(num_layers * hidden_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, 2 * hidden_dim)
            
    def forward(self, x, edge_index, batch):
        aug1, aug2 = self.augmentor
        if type(aug2).__name__ in target_types:
            x1, edge_index1, edge_weight1 = aug1(x, edge_index, batch=batch)
            x2, edge_index2, edge_weight2 = aug2(x, edge_index, batch=batch)
        else:
            x1, edge_index1, edge_weight1 = aug1(x, edge_index)
            x2, edge_index2, edge_weight2 = aug2(x, edge_index)

        if self.save_dir is not None:
            prefix = str(self.prefix)
            self.prefix += 1
            graph1 = Graph(x=x1, edge_index=edge_index1, edge_weights=edge_weight1)
            graph2 = Graph(x=x2, edge_index=edge_index2, edge_weights=edge_weight2)
            pk.dump(graph1, open(os.path.join(self.save_dir, f'{prefix}_aug1.pkl'), 'wb'))
            pk.dump(graph2, open(os.path.join(self.save_dir, f'{prefix}_aug2.pkl'), 'wb'))

        z1 = self.encoder(x1, edge_index1, edge_weight1, batch=batch)
        z2 = self.encoder(x2, edge_index2, edge_weight2, batch=batch)
        return z1, z2

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

##########bgrl_g
class bgrl_GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2, encoder_norm='batch', projector_norm='batch', GNN_model='GPS'):
        super(bgrl_GConv, self).__init__()
        self.activation = torch.nn.PReLU()
        self.dropout = dropout
        self.GNN_model=GNN_model

        self.layers = torch.nn.ModuleList()
        if GNN_model=='GCN':
                    self.layers.append(GCNConv(input_dim, hidden_dim))
        elif GNN_model=='GAT':
                    self.layers.append(GATv2Conv(input_dim, hidden_dim,heads=2,concat=False))
        elif GNN_model=='GPS':
                    self.mlp_0 = MLP([input_dim, hidden_dim, hidden_dim])
                    mlp = MLP([hidden_dim, hidden_dim, hidden_dim])
                    self.layers.append(GPSConv(hidden_dim, GINConv(mlp), heads=2))
        for _ in range(num_layers - 1):
                if GNN_model=='GCN':
                    self.layers.append(GCNConv(hidden_dim, hidden_dim))
                elif GNN_model=='GAT':
                    self.layers.append(GATv2Conv(hidden_dim, hidden_dim,heads=2,concat=False))
                elif GNN_model=='GPS':
                    mlp = MLP([hidden_dim, hidden_dim, hidden_dim])
                    self.layers.append(GPSConv(hidden_dim, GINConv(mlp), heads=2))
                    
        self.batch_norm = Normalize(hidden_dim, norm=encoder_norm)
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            Normalize(hidden_dim, norm=projector_norm),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))

    def forward(self, x, edge_index, edge_weight):
        z = x
        if  self.GNN_model=='GPS':
            z=self.mlp_0(x)
        for conv in self.layers:
            if self.GNN_model=='GCN':
                z = conv(z, edge_index,edge_weight=edge_weight)
                z = self.activation(z)
                z = F.dropout(z, p=self.dropout, training=self.training)
            else:
                z = conv(z, edge_index)   
                z = self.activation(z)
                z = F.dropout(z, p=self.dropout, training=self.training)             
        z = self.batch_norm(z)
        return z, self.projection_head(z)

class bgrl_Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, dropout=0.2, predictor_norm='batch', save_dir=None):
        super(bgrl_Encoder, self).__init__()
        self.online_encoder = encoder
        self.target_encoder = None
        self.augmentor = augmentor
        self.save_dir = save_dir
        self.prefix = 0
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            Normalize(hidden_dim, norm=predictor_norm),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))

    def get_target_encoder(self):
        if self.target_encoder is None:
            self.target_encoder = copy.deepcopy(self.online_encoder)

            for p in self.target_encoder.parameters():
                p.requires_grad = False
        return self.target_encoder

    def update_target_encoder(self, momentum: float):
        for p, new_p in zip(self.get_target_encoder().parameters(), self.online_encoder.parameters()):
            next_p = momentum * p.data + (1 - momentum) * new_p.data
            p.data = next_p

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        aug1, aug2 = self.augmentor
        if  type(aug1).__name__ in target_types:
            x1, edge_index1, edge_weight1 = aug1(x, edge_index,batch=batch)
        else:
            x1, edge_index1, edge_weight1 = aug1(x, edge_index)
        if  type(aug2).__name__ in target_types:
            x2, edge_index2, edge_weight2 = aug2(x, edge_index,batch=batch)
        else:
            x2, edge_index2, edge_weight2 = aug2(x, edge_index)
        if self.save_dir is not None:
            prefix = str(self.prefix)
            self.prefix += 1
            graph1 = Graph(x=x1, edge_index=edge_index1, edge_weights=edge_weight1)
            graph2 = Graph(x=x2, edge_index=edge_index2, edge_weights=edge_weight2)
            pk.dump(graph1, open(os.path.join(self.save_dir, f'{prefix}_aug1.pkl'), 'wb'))
            pk.dump(graph2, open(os.path.join(self.save_dir, f'{prefix}_aug2.pkl'), 'wb'))
        
        h1, h1_online = self.online_encoder(x1, edge_index1, edge_weight=edge_weight1)
        h2, h2_online = self.online_encoder(x2, edge_index2, edge_weight=edge_weight2)

        g1 = global_add_pool(h1, batch)
        h1_pred = self.predictor(h1_online)
        g2 = global_add_pool(h2, batch)
        h2_pred = self.predictor(h2_online)

        with torch.no_grad():
            _, h1_target = self.get_target_encoder()(x1, edge_index1, edge_weight=edge_weight1)
            _, h2_target = self.get_target_encoder()(x2, edge_index2, edge_weight=edge_weight2)
            g1_target = global_add_pool(h1_target, batch)
            g2_target = global_add_pool(h2_target, batch)

        return g1, g2, h1_pred, h2_pred, g1_target, g2_target



def get_store_path(args):
    save_path = os.path.join('grid_search_res', args.method)
    reweight_flag = "_reweight" if args.random_reweight else ""
    same_eps_flag = "_same_setup" if args.same_setup else ""
    if args.same_setup:
        save_path = os.path.join(save_path, 'same_setup')
    os.makedirs(save_path, exist_ok=True)

    if args.perturb_ratio > 0:
        res_path = os.path.join(
            save_path,
            f'{args.encoder}_{args.num_layers}_{args.method}_{args.epochs}epochs_{args.aug_type}_random_search{reweight_flag}{same_eps_flag}_perturb_ratio{args.perturb_ratio}_min_perturb_l2_{args.min_perturb_l2}.jsonl'
        )
    else:
        res_path = os.path.join(
            save_path,
            f'{args.encoder}_{args.num_layers}_{args.method}_{args.epochs}epochs_{args.aug_type}_random_search{reweight_flag}{same_eps_flag}.jsonl'
        )

    return res_path
