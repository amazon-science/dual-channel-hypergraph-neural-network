#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import layer
import torch
import torch.nn as nn
from augmentations import *
from loss import *
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader
from utils import *


class Attention(torch.nn.Module):
    # This class module is a simple attention layer.
    def __init__(self, in_size, hidden_size=64):
        super(Attention, self).__init__()

        self.project = torch.nn.Sequential(
            torch.nn.Linear(in_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, hidden_size, bias=False),
        )

    def forward(self, z):
        w = self.project(z)  # (N, 2, D)
        beta = torch.softmax(w, dim=1)  # (N, 2, D)

        return (beta * z).sum(1), beta  # (N, D), (N, 2, D)


class DualModel(nn.Module):
    """
    Parameters
    ----------
    lr : double
        Learning rate
    dropout : double
        dropout probability for gnn and mlp layers
    gnn_num_layers : int
        number of gnn layers
    mlp_num_layers : int
        number of gnn layers
    *_hidden_channels : int
        dimension of hidden
    num_nodes : int
        number of graph nodes
    num_node_feats : int
        dimension of raw node features
    gnn_encoder_name1 : str
        gnn encoder1 name
    gnn_encoder_name2 : str
        gnn encoder2 name
    predictor_name: str
        link predictor name
    loss_func: str
        loss function name
    optimizer_name: str
        optimization method name
    device: str
        device name: gpu or cpu
    use_node_feats: bool
        whether to use raw node features as input
    train_node_emb: bool
        whether to train node embeddings based on node id
    pretrain_emb: str
        whether to load pretrained node embeddings
    """

    def __init__(
        self,
        lr,
        dropout,
        grad_clip_norm,
        gnn_num_layers,
        mlp_num_layers,
        emb_hidden_channels,
        gnn_hidden_channels,
        mlp_hidden_channels,
        projection_hidden_channels,
        num_nodes,
        num_node_feats,
        gnn_encoder_name1,
        gnn_encoder_name2,
        predictor_name,
        loss_func,
        optimizer_name,
        device,
        use_node_feats,
        train_node_emb,
        pretrain_emb=None,
    ):
        super(DualModel, self).__init__()
        self.loss_func_name = loss_func
        self.num_nodes = num_nodes
        self.num_node_feats = num_node_feats
        self.use_node_feats = use_node_feats
        self.train_node_emb = train_node_emb
        self.clip_norm = grad_clip_norm
        self.out_dim = mlp_hidden_channels
        self.device = device

        # Input Layer for both two channels
        self.input_channels, self.emb = create_input_layer(
            num_nodes=num_nodes,
            num_node_feats=num_node_feats,
            hidden_channels=emb_hidden_channels,
            use_node_feats=use_node_feats,
            train_node_emb=train_node_emb,
            pretrain_emb=pretrain_emb,
        )
        if self.emb is not None:
            self.emb = self.emb.to(device)

        # GNN Layer
        self.encoder1 = create_gnn_layer(
            input_channels=self.input_channels,
            hidden_channels=gnn_hidden_channels,
            num_layers=gnn_num_layers,
            dropout=dropout,
            encoder_name=gnn_encoder_name1,
        ).to(device)

        # HNN Layer
        self.encoder2 = create_gnn_layer(
            input_channels=self.input_channels,
            hidden_channels=gnn_hidden_channels,
            num_layers=gnn_num_layers,
            dropout=dropout,
            encoder_name=gnn_encoder_name2,
        ).to(device)

        # Attention Layer
        self.attention = Attention(in_size=gnn_hidden_channels).to(device)

        # Predict Layer
        self.predictor = create_predictor_layer(
            hidden_channels=mlp_hidden_channels,
            num_layers=mlp_num_layers,
            dropout=dropout,
            predictor_name=predictor_name,
        ).to(device)

        # Projection Head For Contrastive Learning
        self.projection_head = torch.nn.Sequential(
            nn.Linear(mlp_hidden_channels, projection_hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(projection_hidden_channels, projection_hidden_channels),
        ).to(device)

        # Parameters and Optimizer
        self.para_list = (
            list(self.encoder1.parameters())
            + list(self.encoder2.parameters())
            + list(self.projection_head.parameters())
            + list(self.attention.parameters())
            + list(self.predictor.parameters())
        )
        if self.emb is not None:
            self.para_list += list(self.emb.parameters())

        if optimizer_name == "AdamW":
            self.optimizer = torch.optim.AdamW(self.para_list, lr=lr)
        elif optimizer_name == "SGD":
            self.optimizer = torch.optim.SGD(
                self.para_list, lr=lr, momentum=0.9, weight_decay=1e-5, nesterov=True
            )
        else:
            self.optimizer = torch.optim.Adam(self.para_list, lr=lr)

    def forward_cl(self, data1, data2):
        input_feat = self.create_input_feat(data1)
        h1 = self.encoder1(
            input_feat, data1.adj_t
        )  # input_feat & h both have shape: [N_nodes_full, dim]

        if data2.edge_attr is not None:
            h2 = self.encoder2(input_feat, data2.adj_t, data2.edge_attr)
        else:
            h2 = self.encoder2(input_feat, data2.adj_t)

        h = torch.stack([h1, h2], dim=1)
        h, _ = self.attention(h)
        return h

    def param_init(self):
        self.encoder1.reset_parameters()
        self.encoder2.reset_parameters()
        self.predictor.reset_parameters()
        if self.emb is not None:
            torch.nn.init.xavier_uniform_(self.emb.weight)

    def create_input_feat(self, data):
        if self.use_node_feats:
            input_feat = data.x.to(self.device)
            if self.train_node_emb:
                input_feat = torch.cat([self.emb.weight, input_feat], dim=-1)
        else:
            input_feat = self.emb.weight
        return input_feat

    def calculate_loss(self, pos_out, neg_out, num_neg, margin=None):
        if self.loss_func_name == "CE":
            loss = ce_loss(pos_out, neg_out)
        elif self.loss_func_name == "InfoNCE":
            loss = info_nce_loss(pos_out, neg_out, num_neg)
        elif self.loss_func_name == "LogRank":
            loss = log_rank_loss(pos_out, neg_out, num_neg)
        elif self.loss_func_name == "HingeAUC":
            loss = hinge_auc_loss(pos_out, neg_out, num_neg)
        elif self.loss_func_name == "AdaAUC" and margin is not None:
            loss = adaptive_auc_loss(pos_out, neg_out, num_neg, margin)
        elif self.loss_func_name == "WeightedAUC" and margin is not None:
            loss = weighted_auc_loss(pos_out, neg_out, num_neg, margin)
        elif self.loss_func_name == "AdaHingeAUC" and margin is not None:
            loss = adaptive_hinge_auc_loss(pos_out, neg_out, num_neg, margin)
        elif self.loss_func_name == "WeightedHingeAUC" and margin is not None:
            loss = weighted_hinge_auc_loss(pos_out, neg_out, num_neg, margin)
        else:
            loss = auc_loss(pos_out, neg_out, num_neg)
        return loss

    def calculate_cl_loss(self, p1, p2):
        loss = NTXent_loss(p1, p2)
        return loss

    def train_cl(self, data1, data2, split_edge, batch_size, aug_ratio=0.4):
        self.encoder1.train()
        self.encoder2.train()
        self.attention.train()
        self.projection_head.train()
        augmentor = Augmentor(aug_ratio)
        total_loss = total_examples = 0

        # graph augmentation

        # bipartite graph
        data1_edge_index_aug1, data1_edge_index_aug2 = augmentor.apply_aug(
            data1.edge_index
        )
        data1_aug1 = Data(edge_index=data1_edge_index_aug1)
        data1_aug1_adj = SparseTensor(
            row=data1_edge_index_aug1[0],
            col=data1_edge_index_aug1[1],
            sparse_sizes=(self.num_nodes, self.num_nodes),
        )
        data1_aug1.adj = data1_aug1_adj
        data1_aug1.adj_t = data1_aug1_adj.t()

        data1_aug2 = Data(edge_index=data1_edge_index_aug2)
        data1_aug2_adj = SparseTensor(
            row=data1_edge_index_aug2[0],
            col=data1_edge_index_aug2[1],
            sparse_sizes=(self.num_nodes, self.num_nodes),
        )
        data1_aug2.adj = data1_aug2_adj
        data1_aug2.adj_t = data1_aug2_adj.t()

        data1_aug1, data1_aug2 = data1_aug1.to(
            self.device), data1_aug2.to(self.device)

        # hypergraph
        data2_edge_index_aug1, data2_edge_index_aug2 = augmentor.apply_aug(
            data2.edge_index
        )
        data2_aug1 = Data(edge_index=data2_edge_index_aug1)
        data2_aug1.adj_t = data2_edge_index_aug1

        data2_aug2 = Data(edge_index=data2_edge_index_aug2)
        data2_aug2.adj_t = data2_edge_index_aug2

        data2_aug1, data2_aug2 = data2_aug1.to(
            self.device), data2_aug2.to(self.device)

        pos_train_edge, _ = get_pos_neg_edges("train", split_edge)

        train_nodes = torch.unique(pos_train_edge.reshape((-1,)))

        for perm in DataLoader(range(train_nodes.shape[0]), batch_size, shuffle=False):
            self.optimizer.zero_grad()
            h1 = self.forward_cl(data1_aug1, data2_aug1)
            h2 = self.forward_cl(data1_aug2, data2_aug2)

            nodes = train_nodes[perm]

            out1 = self.projection_head(h1[nodes])
            out2 = self.projection_head(h2[nodes])

            if self.loss_func_name == "CL":
                loss = self.calculate_cl_loss(out1, out2)
            else:
                raise NotImplementedError(
                    f"loss {self.loss_func_name} not implemented")

            loss.backward()

            if self.clip_norm >= 0:
                torch.nn.utils.clip_grad_norm_(
                    self.encoder1.parameters(), self.clip_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.encoder2.parameters(), self.clip_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.attention.parameters(), self.clip_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.projection_head.parameters(), self.clip_norm
                )

            self.optimizer.step()

            total_loss += loss.item() * batch_size
            total_examples += batch_size

        return total_loss / total_examples

    def train(self, data1, data2, split_edge, batch_size, num_neg, use_drop_edge):
        self.encoder1.train()
        self.encoder2.train()
        self.attention.train()
        self.predictor.train()

        # whether to use DropEdge
        if use_drop_edge:
            # bipartite graph
            data1_edge_index_aug = drop_edges(data1.edge_index)
            data1_aug = Data(edge_index=data1_edge_index_aug)
            data1_aug_adj = SparseTensor(
                row=data1_edge_index_aug[0],
                col=data1_edge_index_aug[1],
                sparse_sizes=(self.num_nodes, self.num_nodes),
            )
            data1_aug.adj = data1_aug_adj
            data1_aug.adj_t = data1_aug_adj.t()

            # hypergraph
            data2_edge_index_aug = drop_edges(data2.edge_index)
            data2_aug = Data(edge_index=data2_edge_index_aug)
            data2_aug.adj_t = data2_edge_index_aug

            data1_aug, data2_aug = data1_aug.to(
                self.device), data2_aug.to(self.device)

        pos_train_edge, neg_train_edge = get_pos_neg_edges("train", split_edge)

        if "weight" in split_edge["train"]:
            edge_weight_margin = split_edge["train"]["weight"].to(self.device)
        else:
            edge_weight_margin = None

        total_loss = total_examples = 0

        for perm in DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=True):
            self.optimizer.zero_grad()

            input_feat = self.create_input_feat(data1)
            if use_drop_edge:
                h1 = self.encoder1(
                    input_feat, data1_aug.adj_t
                )  # input_feat & h both have shape: [N_nodes_full, dim=64]

                h2 = self.encoder2(input_feat, data2_aug.adj_t)
            else:
                h1 = self.encoder1(
                    input_feat, data1.adj_t
                )  # input_feat & h both have shape: [N_nodes_full, dim=64]

                h2 = self.encoder2(input_feat, data2.adj_t)

            h = torch.stack([h1, h2], dim=1)
            h, _ = self.attention(h)

            pos_edge = pos_train_edge[perm].t().to(self.device)
            neg_edge = torch.reshape(
                neg_train_edge[perm], (-1, 2)).t().to(self.device)

            pos_out = self.predictor(
                h[pos_edge[0]], h[pos_edge[1]]
            )  # shape: [num_pos_edges]
            neg_out = self.predictor(h[neg_edge[0]], h[neg_edge[1]])

            weight_margin = (
                edge_weight_margin[perm] if edge_weight_margin is not None else None
            )

            loss = self.calculate_loss(
                pos_out, neg_out, num_neg, margin=weight_margin)
            loss.backward()

            if self.clip_norm >= 0:
                torch.nn.utils.clip_grad_norm_(
                    self.encoder1.parameters(), self.clip_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.encoder2.parameters(), self.clip_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.attention.parameters(), self.clip_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.predictor.parameters(), self.clip_norm
                )

            self.optimizer.step()

            num_examples = pos_out.size(0)
            total_loss += loss.item() * num_examples
            total_examples += num_examples

        return total_loss / total_examples, h

    @torch.no_grad()
    def batch_predict(self, h, edges, batch_size):
        # this function take the entire split of edges, use the model predictor head, and output the score for the entire split of edges. Used only in test, not during training.
        # # h has shape: [N_total_nodes, dim] ;
        # edges shape: [N_total_edge_pos_or_neg, 2], pred shape: [N_total_edge_pos_or_neg]
        preds = []
        for perm in DataLoader(range(edges.size(0)), batch_size):
            edge = edges[perm].t()
            preds += [self.predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        pred = torch.cat(preds, dim=0)
        return pred

    @torch.no_grad()
    def test(self, data1, data2, split_edge, batch_size, evaluator, eval_metric):
        self.encoder1.eval()
        self.encoder2.eval()
        self.attention.eval()
        self.predictor.eval()

        input_feat = self.create_input_feat(data1)
        h1 = self.encoder1(
            input_feat, data1.adj_t
        )  # input_feat & h both have shape: [N_nodes_full, dim=64]

        h2 = self.encoder2(input_feat, data2.adj_t)

        h = torch.stack([h1, h2], dim=1)
        h, _ = self.attention(h)

        # validation set
        pos_val_edge, neg_val_edge = get_pos_neg_edges("val", split_edge)
        pos_val_pred = self.batch_predict(h, pos_val_edge, batch_size)
        neg_val_pred = self.batch_predict(h, neg_val_edge, batch_size)

        # testing set
        pos_head_edge, neg_head_edge = get_pos_neg_edges("head", split_edge)
        pos_tail1_edge, neg_tail1_edge = get_pos_neg_edges("tail1", split_edge)
        pos_tail2_edge, neg_tail2_edge = get_pos_neg_edges("tail2", split_edge)
        pos_iso_edge, neg_iso_edge = get_pos_neg_edges("iso", split_edge)

        pos_head_edge, neg_head_edge = pos_head_edge.to(
            self.device), neg_head_edge.to(self.device)
        pos_tail1_edge, neg_tail1_edge = pos_tail1_edge.to(
            self.device), neg_tail1_edge.to(self.device)
        pos_tail2_edge, neg_tail2_edge = pos_tail2_edge.to(
            self.device), neg_tail2_edge.to(self.device)
        pos_iso_edge, neg_iso_edge = pos_iso_edge.to(
            self.device), neg_iso_edge.to(self.device)

        pos_head_pred = self.batch_predict(h, pos_head_edge, batch_size)
        neg_head_pred = self.batch_predict(h, neg_head_edge, batch_size)

        pos_tail1_pred = self.batch_predict(h, pos_tail1_edge, batch_size)
        neg_tail1_pred = self.batch_predict(h, neg_tail1_edge, batch_size)

        pos_tail2_pred = self.batch_predict(h, pos_tail2_edge, batch_size)
        neg_tail2_pred = self.batch_predict(h, neg_tail2_edge, batch_size)

        pos_iso_pred = self.batch_predict(h, pos_iso_edge, batch_size)
        neg_iso_pred = self.batch_predict(h, neg_iso_edge, batch_size)

        if eval_metric == "hits":
            results = evaluate_hits(
                evaluator,
                pos_val_pred,
                neg_val_pred,
                pos_head_pred,
                neg_head_pred,
                pos_tail1_pred,
                neg_tail1_pred,
                pos_tail2_pred,
                neg_tail2_pred,
                pos_iso_pred,
                neg_iso_pred,
            )
        elif eval_metric == "mrr":
            results = evaluate_mrr(
                evaluator,
                pos_val_pred,
                neg_val_pred,
                pos_head_pred,
                neg_head_pred,
                pos_tail1_pred,
                neg_tail1_pred,
                pos_tail2_pred,
                neg_tail2_pred,
                pos_iso_pred,
                neg_iso_pred,
            )

        elif "recall_my" in eval_metric:
            results = evaluate_recall_my(
                pos_val_pred,
                neg_val_pred,
                pos_head_pred,
                neg_head_pred,
                pos_tail1_pred,
                neg_tail1_pred,
                pos_tail2_pred,
                neg_tail2_pred,
                pos_iso_pred,
                neg_iso_pred,
                topk=eval_metric.split("@")[1],
            )

        return results, h

    @torch.no_grad()
    def get_attention(self, data1, data2, split_edge, batch_size):
        self.encoder1.eval()
        self.encoder2.eval()
        self.attention.eval()
        self.predictor.eval()

        input_feat = self.create_input_feat(data1)
        h1 = self.encoder1(
            input_feat, data1.adj_t
        )  # input_feat & h both have shape: [N_nodes_full, dim=64]

        h2 = self.encoder2(input_feat, data2.adj_t)

        h = torch.stack([h1, h2], dim=1)
        h, beta = self.attention(h)

        pos_head_edge, _ = get_pos_neg_edges("head", split_edge)
        pos_tail1_edge, _ = get_pos_neg_edges("tail1", split_edge)
        pos_tail2_edge, _ = get_pos_neg_edges("tail2", split_edge)
        pos_iso_edge, _ = get_pos_neg_edges("iso", split_edge)

        pos_head_edge = pos_head_edge.to(self.device)
        pos_tail1_edge = pos_tail1_edge.to(self.device)
        pos_tail2_edge = pos_tail2_edge.to(self.device)
        pos_iso_edge = pos_iso_edge.to(self.device)

        head_attention = torch.cat(
            (beta[pos_head_edge.T[0]], beta[pos_head_edge.T[1]])).squeeze()
        tail1_attention = torch.cat(
            (beta[pos_tail1_edge.T[0]], beta[pos_tail1_edge.T[1]])).squeeze()
        tail2_attention = torch.cat(
            (beta[pos_tail2_edge.T[0]], beta[pos_tail2_edge.T[1]])).squeeze()
        iso_attention = torch.cat(
            (beta[pos_iso_edge.T[0]], beta[pos_iso_edge.T[1]])).squeeze()

        attention_results = {
            "head": head_attention,
            "tail1": tail1_attention,
            "tail2": tail2_attention,
            "iso": iso_attention,
        }

        return attention_results


def create_input_layer(num_nodes, num_node_feats, hidden_channels, use_node_feats=True, train_node_emb=False, pretrain_emb=None):
    emb = None
    if use_node_feats:
        input_dim = num_node_feats
        if train_node_emb:
            emb = torch.nn.Embedding(num_nodes, hidden_channels)
            input_dim += hidden_channels
        elif pretrain_emb is not None and pretrain_emb != "":
            weight = torch.load(pretrain_emb)
            emb = torch.nn.Embedding.from_pretrained(weight)
            input_dim += emb.weight.size(1)
    else:
        if pretrain_emb is not None and pretrain_emb != "":
            weight = torch.load(pretrain_emb)
            emb = torch.nn.Embedding.from_pretrained(weight)
            input_dim = emb.weight.size(1)
        else:
            emb = torch.nn.Embedding(num_nodes, hidden_channels)
            input_dim = hidden_channels
    return input_dim, emb


def create_gnn_layer(input_channels, hidden_channels, num_layers, dropout=0, encoder_name="SAGE"):
    if encoder_name.upper() == "GCN":
        return layer.GCN(
            input_channels, hidden_channels, hidden_channels, num_layers, dropout
        )
    elif encoder_name.upper() == "GAT":
        return layer.GAT(
            input_channels, hidden_channels, hidden_channels, num_layers, dropout
        )
    elif encoder_name.upper() == "WSAGE":
        return layer.WSAGE(
            input_channels, hidden_channels, hidden_channels, num_layers, dropout
        )
    elif encoder_name.upper() == "TRANSFORMER":
        return layer.Transformer(
            input_channels, hidden_channels, hidden_channels, num_layers, dropout
        )
    elif encoder_name.upper() == "SAGE":
        return layer.SAGE(
            input_channels, hidden_channels, hidden_channels, num_layers, dropout
        )
    elif encoder_name.upper() == "MLP":
        return layer.MLP(
            input_channels, hidden_channels, hidden_channels, num_layers, dropout
        )
    elif encoder_name.upper() == "HYPERGCN":
        return layer.HyperGCN(
            input_channels, hidden_channels, hidden_channels, num_layers, dropout
        )
    else:
        raise NotImplementedError(
            f"encoder_name {encoder_name} not implemented")


def create_predictor_layer(hidden_channels, num_layers, dropout=0, predictor_name="MLP"):
    predictor_name = predictor_name.upper()
    if predictor_name == "DOT":
        return layer.DotPredictor()
    elif predictor_name == "BIL":
        return layer.BilinearPredictor(hidden_channels)
    elif predictor_name == "MLP":
        return layer.MLPPredictor(hidden_channels, hidden_channels, 1, num_layers, dropout)
    elif predictor_name == "MLPDOT":
        return layer.MLPDotPredictor(hidden_channels, 1, num_layers, dropout)
    elif predictor_name == "MLPBIL":
        return layer.MLPBilPredictor(hidden_channels, 1, num_layers, dropout)
    elif predictor_name == "MLPCAT":
        return layer.MLPCatPredictor(hidden_channels, hidden_channels, 1, num_layers, dropout)
