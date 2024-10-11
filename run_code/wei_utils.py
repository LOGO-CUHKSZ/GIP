import utils 
import torch
from torch.optim import Adam
from torch_geometric.data import DataLoader
from GCL.eval import from_predefined_split
from abc import ABC, abstractmethod
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import umap
import torch
from sklearn.svm import SVC  
from sklearn.model_selection import train_test_split
import os
import pickle

import GCL.augmentors as A
from torch_geometric.utils import get_laplacian, to_dense_adj
from tqdm import tqdm
from torch.optim import Adam
from GCL.models.contrast_model import WithinEmbedContrast, DualBranchContrast
from GCL.models import BootstrapContrast
import numpy as np
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import GCL.losses as L
from torch_geometric.utils import dropout_adj, to_networkx, to_undirected, degree, to_scipy_sparse_matrix, \
    from_scipy_sparse_matrix, sort_edge_index, add_self_loops, subgraph

def get_split(num_samples: int, train_ratio: float = 0.1, test_ratio: float = 0.8, seed=10):
    assert train_ratio + test_ratio < 1
    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)
    indices = torch.randperm(num_samples)
    return {
        'train': indices[:train_size],
        'test': indices[train_size: test_size + train_size],
        'valid': indices[test_size + train_size:]
    }

def load_data(data_info):
    dataset = utils.get_dataset(data_info['path'],data_info['name'])
    data = dataset[0]

    return data

def load_dataset(data_info):
    dataset= utils.get_dataset(data_info['path'],data_info['name'])

    return dataset

###gbt_graph level
def GBTG_train(encoder_model, contrast_model, dataloader, optimizer):
    encoder_model.train()
    epoch_loss = 0
    for data in dataloader:
        data = data.to('cuda')
        optimizer.zero_grad()

        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        z1,z2 = encoder_model(data.x, data.edge_index, data.batch)
        loss = contrast_model(z1, z2)
     
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss

def load_graphs(dir):
    def get_index(file_name):
        return int(file_name.split('_')[0])
    graph_files = os.listdir(dir)
    aug1_files = [file_name for file_name in graph_files if 'aug1' in file_name]
    aug2_files = [file_name for file_name in graph_files if 'aug2' in file_name]

    aug1_files = sorted(aug1_files, key=get_index)
    aug2_files = sorted(aug2_files, key=get_index)

    aug1_graphs = []
    aug2_graphs = []
    for aug1_file, aug2_file in zip(aug1_files, aug2_files):
        aug1_graphs.append(pickle.load(open(os.path.join(dir, aug1_file), 'rb')))
        aug2_graphs.append(pickle.load(open(os.path.join(dir, aug2_file), 'rb')))

    return aug1_graphs, aug2_graphs

def G_test(encoder_model, dataloader):
    encoder_model.eval()
    x = []
    y = []
    for data in dataloader:
        data = data.to('cuda')
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        g1, g2 = encoder_model(data.x, data.edge_index, data.batch)
        x.append(g1 + g2)
        y.append(data.y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)
    split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.1)
    result = utils.LREvaluator()(x, y, split)
    return result


def accuracy_eval_gbtg(
        dataset, 
        aug1, 
        aug2, 
        num_layers,
        device='cuda:0',
        dim = 256, 
        lr = 0.001, 
        epoch = 200,
        mode='L2L',
        batch_size=128,
        graph_save_dir=None,
        encoder='GCN'
    ):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    gconv = utils.G_GConv(input_dim=dataset.num_features, hidden_dim=dim, num_layers=num_layers,GNN_model=encoder).to(device)
    encoder_model = utils.GBTG_Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=dim, save_dir=graph_save_dir).to(device)
    contrast_model = WithinEmbedContrast(loss=L.BarlowTwins()).to(device)
    optimizer = Adam(encoder_model.parameters(), lr=lr)
    best_test_result=0
    best_epoch=[]
    result_dict=None
    for i in tqdm(range(1, epoch + 1)):
                    loss = GBTG_train(encoder_model, contrast_model,dataloader, optimizer)
                    if i % 20==0 or i == epoch:
                        test_result = G_test(encoder_model,dataloader)
                        if test_result['accuracy']> best_test_result:
                            best_test_result= test_result['accuracy']   
                            result_dict = test_result 
                                 
    return result_dict

###MVGRL-G
def MVGRL_train(encoder_model, contrast_model, dataloader, optimizer):
    encoder_model.train()
    epoch_loss = 0
    for data in dataloader:
        # print(data)
        data = data.to('cuda')
        optimizer.zero_grad()

        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        h1, h2, g1, g2 = encoder_model(data.x, data.edge_index, data.batch)
        loss = contrast_model(h1=h1, h2=h2, g1=g1, g2=g2, batch=data.batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss

def MVGRL_test(encoder_model, dataloader):
    encoder_model.eval()
    x = []
    y = []
    for data in dataloader:
        data = data.to('cuda')
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        _, _, g1, g2 = encoder_model(data.x, data.edge_index, data.batch)
        x.append(g1 + g2)
        y.append(data.y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)    
    split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.1)
    result = utils.LREvaluator()(x, y, split)
    return result


def accuracy_eval_MVGRLG(
        dataset, 
        aug1, 
        aug2, 
        num_layers,
        device='cuda:0',
        dim = 256, 
        lr = 0.001, 
        epoch = 200,
        mode='L2L',
        batch_size=128,
        graph_save_dir=None,
        encoder='GCN'
    ):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    gconv1 = utils.MVGRL_GConv(input_dim=dataset.num_features, hidden_dim=dim, num_layers=num_layers,GNN_model=encoder).to(device)
    gconv2 = utils.MVGRL_GConv(input_dim=dataset.num_features, hidden_dim=dim, num_layers=num_layers,GNN_model=encoder).to(device)
    mlp1 = utils.FC(input_dim=dim, output_dim=dim)
    mlp2 = utils.FC(input_dim=dim*num_layers, output_dim=dim)
    encoder_model = utils.MVGRL_Encoder(gcn1=gconv1, gcn2=gconv2, mlp1=mlp1, mlp2=mlp2,aug1=aug1, aug2=aug2, save_dir=graph_save_dir).to(device)
    contrast_model = DualBranchContrast(loss=L.JSD(), mode='G2L').to(device)    
    optimizer = Adam(encoder_model.parameters(), lr=lr)
    best_test_result=0
    best_epoch=[]
    result_dict=None
    for i in tqdm(range(1, epoch + 1)):
                    loss = MVGRL_train(encoder_model, contrast_model,dataloader, optimizer)
                    if i % 50==0 or i == epoch:
                        test_result = MVGRL_test(encoder_model,dataloader)
                        if test_result['accuracy']> best_test_result:
                            best_test_result= test_result['accuracy']   
                            result_dict = test_result      
    return result_dict
#####grace
def grace_train(encoder_model, contrast_model, dataloader, optimizer):
    encoder_model.train()
    epoch_loss = 0
    for data in dataloader:
        data = data.to('cuda')
        optimizer.zero_grad()

        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        z1,z2 = encoder_model(data.x, data.edge_index, data.batch)
        h1, h2 = [encoder_model.project(x) for x in [z1, z2]]   
        loss = contrast_model(g1=h1, g2=h2,batch=data.batch)
     
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss

def accuracy_eval_graceg(
        dataset, 
        aug1, 
        aug2, 
        num_layers,
        device='cuda:0',
        dim = 256, 
        lr = 0.001, 
        epoch = 200,
        mode='L2L',
        batch_size=128,
        graph_save_dir=None,
        encoder='GCN',
        stage='all'
    ):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    gconv = utils.G_GConv(input_dim=dataset.num_features, hidden_dim=dim, num_layers=num_layers,GNN_model=encoder).to(device)
    encoder_model = utils.Grace_Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=512, proj_dim=128, save_dir=graph_save_dir,num_layers=num_layers).to(device)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G', intraview_negs=True).to(device)
    optimizer = Adam(encoder_model.parameters(), lr=lr)
    best_test_result=0
    result_dict=None
    for i in tqdm(range(1, 1 + epoch)):
                    loss = grace_train(encoder_model, contrast_model,dataloader, optimizer)
                    if i % 50==1 or i == epoch:
                        test_result = G_test(encoder_model,dataloader)
                        if test_result['accuracy']> best_test_result:
                            best_test_result= test_result['accuracy']   
                            result_dict = test_result
    return result_dict

#####bgrl
def bgrlg_train(encoder_model, contrast_model, dataloader, optimizer):
    encoder_model.train()
    total_loss = 0

    for data in dataloader:
        data = data.to('cuda')
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32).to(data.batch.device)

        optimizer.zero_grad()
        _, _, h1_pred, h2_pred, g1_target, g2_target = encoder_model(data.x, data.edge_index, batch=data.batch)

        loss = contrast_model(h1_pred=h1_pred, h2_pred=h2_pred,
                              g1_target=g1_target.detach(), g2_target=g2_target.detach(), batch=data.batch)
        loss.backward()
        optimizer.step()
        encoder_model.update_target_encoder(0.99)
        total_loss += loss.item()
    return total_loss


def bgrlg_test(encoder_model, dataloader):
    encoder_model.eval()
    x = []
    y = []
    for data in dataloader:
        data = data.to('cuda')
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        g1, g2, _, _, _, _ = encoder_model(data.x, data.edge_index, batch=data.batch)
        z = torch.cat([g1, g2], dim=1)
        x.append(z)
        y.append(data.y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)

    split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.1)
    result = utils.LREvaluator()(x, y, split)
    return result

def accuracy_eval_bgrlg(
        dataset, 
        aug1, 
        aug2, 
        num_layers,
        device='cuda:0',
        dim = 256, 
        lr = 0.001, 
        epoch = 200,
        mode='L2L',
        batch_size=128,
        graph_save_dir=None,
        encoder='GCN'
    ):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    gconv = utils.bgrl_GConv(input_dim=dataset.num_features, hidden_dim=dim, num_layers=num_layers,GNN_model=encoder).to(device)
    encoder_model = utils.bgrl_Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=dim, save_dir=graph_save_dir).to(device)
    contrast_model = BootstrapContrast(loss=L.BootstrapLatent(), mode='G2L').to(device)

    optimizer = Adam(encoder_model.parameters(), lr=lr)
    best_test_result=0
    best_epoch=[]
    result_dict=None
    for i in tqdm(range(1, 1 + epoch)):
                    loss = bgrlg_train(encoder_model, contrast_model,dataloader, optimizer)
                    # pbar.set_postfix({'loss': loss})
                    # pbar.update()
                    if i % 50==0 or i == epoch:
                        test_result = bgrlg_test(encoder_model,dataloader)
                        if test_result['accuracy']> best_test_result:
                            best_test_result= test_result['accuracy']   
                            result_dict = test_result      
    return result_dict



def get_augmentor(params, eps):
    if params['aug_type'] == 'EdgeSparse':
        return edge_sparse_augmentor_cpu(
            params['dataset'][0],
            eps,
            params['Anorm'],
            params['Lnorm'],
            reweight=params['reweight']
        )
    
    if params['aug_type'] == 'EdgeRemoving':
        return A.EdgeRemoving(
            pe=eps
        )
    
    if params['aug_type'] == 'EdgeAdding':
        return A.EdgeAdding(
            pe=eps
        )
    
    if params['aug_type'] == 'Identity':
        return A.Identity()


##
#######graph level task
from torch_geometric.utils import unbatch, unbatch_edge_index
from torch_geometric.data import Batch, Data
from torch_sparse import SparseTensor, coalesce
from typing import Optional, Tuple, NamedTuple, List

def coalesce_edge_index(edge_index: torch.Tensor, edge_weights: Optional[torch.Tensor] = None) -> (torch.Tensor, torch.FloatTensor):
    num_edges = edge_index.size()[1]
    num_nodes = edge_index.max().item() + 1
    edge_weights = edge_weights if edge_weights is not None else torch.ones((num_edges,), dtype=torch.float32, device=edge_index.device)

    return coalesce(edge_index, edge_weights, m=num_nodes, n=num_nodes) 
class Graph(NamedTuple):
    x: torch.FloatTensor
    edge_index: torch.LongTensor
    edge_weights: Optional[torch.FloatTensor]

    def unfold(self) -> Tuple[torch.FloatTensor, torch.LongTensor, Optional[torch.FloatTensor]]:
        return self.x, self.edge_index, self.edge_weights
    

class Augmentor(ABC):
    """Base class for graph augmentors."""
    def __init__(self):
        pass

    @abstractmethod
    def augment(self, g: Graph, batch: torch.Tensor) -> Graph:
        raise NotImplementedError(f"GraphAug.augment should be implemented.")
    def __call__(
            self, x: torch.FloatTensor,
            edge_index: torch.LongTensor, edge_weight: Optional[torch.FloatTensor] = None,
            batch = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return self.augment(Graph(x, edge_index, edge_weight),batch).unfold()

class Edge_Adding_pergraph(Augmentor):
    def __init__(self, pe: float):
        super(Edge_Adding_pergraph, self).__init__()
        self.pe = pe  # Edge addition probability within each graph

    def add_edges(self, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        device = edge_index.device
        unique_graphs, graph_sizes = torch.unique(batch, return_counts=True)
        num_graphs = unique_graphs.size(0)

        # Compute cumulative sum of graph sizes
        cum_sizes = torch.cat([torch.tensor([0], device=device), graph_sizes.cumsum(0)])

        # Calculate average number of edges to add per graph
        avg_edges_to_add = int(self.pe * edge_index.size(1) / num_graphs)
        total_new_edges = avg_edges_to_add * num_graphs

        # if total_new_edges == 0:
        #     return edge_index

        # Create a range for each graph
        graph_ranges = torch.arange(num_graphs, device=device).repeat_interleave(avg_edges_to_add)

        # Generate offsets for each new edge
        offsets = cum_sizes[graph_ranges]

        # Generate random node indices within each graph
        max_nodes = graph_sizes[graph_ranges]
        src_nodes = torch.rand(total_new_edges, device=device) * max_nodes
        dst_nodes = torch.rand(total_new_edges, device=device) * max_nodes
        
        src_nodes = src_nodes.long()
        dst_nodes = dst_nodes.long()

        # Add offsets to get global node indices
        src_nodes += offsets
        dst_nodes += offsets
        # Stack new edges
        new_edges = torch.stack([src_nodes, dst_nodes], dim=0)

        # Concatenate with original edges
        edge_index = torch.cat([edge_index, new_edges], dim=1)
        edge_index = sort_edge_index(edge_index)
        edge_index = coalesce_edge_index(edge_index)[0]
        return edge_index

    def augment(self, batch_data: Data, batch: torch.Tensor) -> Data:
        x, edge_index = batch_data.x, batch_data.edge_index
        edge_index = self.add_edges(edge_index, batch)
        return Graph(x=x, edge_index=edge_index, edge_weights=None)


class GIP_S(Augmentor):
    def __init__(self, pe: float):
        super(GIP_S, self).__init__()
        self.pe = pe
        
    def add_edge(self, edge_index: torch.Tensor, batch: torch.Tensor, ratio: float) -> torch.Tensor:
        num_edges = edge_index.size(1)
        num_add = int(num_edges * ratio)

        unique_graphs = batch.unique()
        graph_node_indices = [torch.where(batch == graph_id)[0] for graph_id in unique_graphs]
        graph_node_indices = torch.cat(graph_node_indices)

        src_graphs = torch.randint(0, len(unique_graphs), (num_add,))
        dst_graphs = torch.randint(0, len(unique_graphs), (num_add,))

        different_graph_mask = src_graphs != dst_graphs
        while not different_graph_mask.all():
            dst_graphs[~different_graph_mask] = torch.randint(0, len(unique_graphs), (dst_graphs[~different_graph_mask].size(0),))
            different_graph_mask = src_graphs != dst_graphs

        src_nodes = torch.randint(0, graph_node_indices.size(0), (num_add,))
        dst_nodes = torch.randint(0, graph_node_indices.size(0), (num_add,))

        new_edges = torch.stack([graph_node_indices[src_nodes], graph_node_indices[dst_nodes]], dim=0)

        edge_index = torch.cat([edge_index, new_edges.to(edge_index.device)], dim=1)

        return edge_index


    def augment(self, batch_data: Data, batch: torch.Tensor) -> Data:
        x, edge_index = batch_data.x, batch_data.edge_index
        edge_index = self.add_edge(edge_index, batch, ratio=self.pe)
        return Graph(x=x, edge_index=edge_index, edge_weights=None)


class GIP_randomwalk(Augmentor):
    def __init__(self, pe: float,k : int):
        super(GIP_randomwalk, self).__init__()
        self.pe = pe
        
        self.k = k
        
    def add_edge(self, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        num_nodes = int(edge_index.max()) + 1
        num_add = int(edge_index.size(1) * self.pe) // self.k  
        new_edges = torch.empty((2, num_add * (self.k - 1)), dtype=torch.long, device=edge_index.device)
        for i in range(num_add):
            selected_nodes = torch.randint(0, num_nodes, (self.k,), device=edge_index.device)            
            new_edges[0, i * (self.k - 1):(i + 1) * (self.k - 1)] = selected_nodes[:-1]
            new_edges[1, i * (self.k - 1):(i + 1) * (self.k - 1)] = selected_nodes[1:]
        edge_index = torch.cat([edge_index, new_edges], dim=1)
        return edge_index


    def augment(self, batch_data: Data, batch: torch.Tensor) -> Data:
        x, edge_index = batch_data.x, batch_data.edge_index
        edge_index = self.add_edge(edge_index, batch)
        return Graph(x=x, edge_index=edge_index, edge_weights=None)

    

#############reset augmentor
from GCL.augmentors.augmentor import Graph, Augmentor

class GIP(Augmentor):
    def __init__(self, pe: float):
        super(GIP, self).__init__()
        self.pe = pe
    def add_edge(self, edge_index: torch.Tensor, ratio: float) -> torch.Tensor:
        num_edges = edge_index.size()[1]
        num_nodes = edge_index.max().item() + 1
        num_add = int(num_edges * ratio)

        new_edge_index = torch.randint(0, num_nodes - 1, size=(2, num_add)).to(edge_index.device)
        edge_index = torch.cat([edge_index, new_edge_index], dim=1)
        edge_index = sort_edge_index(edge_index)
        edge_index = coalesce_edge_index(edge_index)[0] 
        return edge_index

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        edge_index = self.add_edge(edge_index, ratio=self.pe)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)
##
class Graph_ER(NamedTuple):
    x: torch.FloatTensor
    edge_index: torch.LongTensor
    edge_weights: Optional[torch.FloatTensor]
    batch : torch.LongTensor

    def unfold(self) -> Tuple[torch.FloatTensor, torch.LongTensor, Optional[torch.FloatTensor],torch.LongTensor]:
        return self.x, self.edge_index, self.edge_weights,self.batch
    
class Augmentor_Latent_ER(ABC):
    """Base class for graph augmentors."""
    def __init__(self):
        pass

    @abstractmethod
    def augment(self, g: Graph, batch: torch.Tensor) -> Graph:
        raise NotImplementedError(f"GraphAug.augment should be implemented.")
    def __call__(
            self, x: torch.FloatTensor,
            edge_index: torch.LongTensor, edge_weight: Optional[torch.FloatTensor] = None,
            batch = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return self.augment(Graph(x, edge_index, edge_weight),batch).unfold()