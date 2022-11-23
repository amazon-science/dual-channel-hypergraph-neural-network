#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import layer
import torch.nn as nn
from augmentations import *
from loss import *
from torch.utils.data import DataLoader
from utils import *


class BaseModel(nn.Module):
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
    gnn_encoder_name : str
        gnn encoder name
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
        gnn_encoder_name,
        predictor_name,
        loss_func,
        optimizer_name,
        device,
        use_node_feats,
        train_node_emb,
        pretrain_emb=None,
    ):
        super(BaseModel, self).__init__()
        self.encoder_name = gnn_encoder_name
        self.loss_func_name = loss_func
        self.num_nodes = num_nodes
        self.num_node_feats = num_node_feats
        self.use_node_feats = use_node_feats
        self.train_node_emb = train_node_emb
        self.clip_norm = grad_clip_norm
        self.device = device

        # Input Layer
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
        self.encoder = create_gnn_layer(
            input_channels=self.input_channels,
            hidden_channels=gnn_hidden_channels,
            num_layers=gnn_num_layers,
            dropout=dropout,
            encoder_name=gnn_encoder_name,
        ).to(device)

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
            list(self.encoder.parameters())
            + list(self.predictor.parameters())
            + list(self.projection_head.parameters())
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

    def forward_cl(self, data):
        input_feat = self.create_input_feat(data)
        if data.edge_attr is not None:
            h = self.encoder(input_feat, data.adj_t, data.edge_attr)
        else:
            h = self.encoder(input_feat, data.adj_t)
        return h

    def param_init(self):
        self.encoder.reset_parameters()
        self.predictor.reset_parameters()
        if self.emb is not None:
            torch.nn.init.xavier_uniform_(self.emb.weight)

    def create_input_feat(self, data):
        if self.use_node_feats:
            if self.encoder_name.upper() == "MLP":
                input_feat = self.embs.to(self.device)
            else:
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

    def train_cl(self, data, split_edge, batch_size, aug_ratio=0.4):
        self.encoder.train()
        self.projection_head.train()
        augmentor = Augmentor(aug_ratio)
        total_loss = total_examples = 0

        # graph augmentation

        # bipartite graph
        if "hyper" not in self.encoder_name:
            data_edge_index_aug1, data_edge_index_aug2 = augmentor.apply_aug(
                data.edge_index
            )
            data_aug1 = Data(edge_index=data_edge_index_aug1)
            data_aug1_adj = SparseTensor(
                row=data_edge_index_aug1[0],
                col=data_edge_index_aug1[1],
                sparse_sizes=(self.num_nodes, self.num_nodes),
            )
            data_aug1.adj = data_aug1_adj
            data_aug1.adj_t = data_aug1_adj.t()

            data_aug2 = Data(edge_index=data_edge_index_aug2)
            data_aug2_adj = SparseTensor(
                row=data_edge_index_aug2[0],
                col=data_edge_index_aug2[1],
                sparse_sizes=(self.num_nodes, self.num_nodes),
            )
            data_aug2.adj = data_aug2_adj
            data_aug2.adj_t = data_aug2_adj.t()

            data_aug1, data_aug2 = data_aug1.to(
                self.device), data_aug2.to(self.device)
        else:
            # hypergraph
            data_edge_index_aug1, data_edge_index_aug2 = augmentor.apply_aug(
                data.edge_index
            )
            data_aug1 = Data(edge_index=data_edge_index_aug1)
            data_aug1.adj_t = data_edge_index_aug1

            data_aug2 = Data(edge_index=data_edge_index_aug2)
            data_aug2.adj_t = data_edge_index_aug2

            data_aug1, data_aug2 = data_aug1.to(
                self.device), data_aug2.to(self.device)

        pos_train_edge, _ = get_pos_neg_edges("train", split_edge)

        train_nodes = torch.unique(pos_train_edge.reshape((-1,)))

        for perm in DataLoader(range(train_nodes.shape[0]), batch_size, shuffle=False):
            self.optimizer.zero_grad()

            h1 = self.forward_cl(data_aug1)
            h2 = self.forward_cl(data_aug2)

            nodes = train_nodes[perm]

            out1 = self.projection_head(h1[nodes])
            out2 = self.projection_head(h2[nodes])

            if self.loss_func_name == "CL":
                loss = self.calculate_cl_loss(out1, out2)

            loss.backward()

            if self.clip_norm >= 0:
                torch.nn.utils.clip_grad_norm_(
                    self.encoder.parameters(), self.clip_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.projection_head.parameters(), self.clip_norm
                )

            self.optimizer.step()

            total_loss += loss.item() * batch_size
            total_examples += batch_size

        return total_loss / total_examples

    def train(self, data, split_edge, batch_size, num_neg, use_drop_edge):
        self.encoder.train()
        self.predictor.train()

        # whether to use DropEdge

        if use_drop_edge:
            # bipartite graph
            if "hyper" not in self.encoder_name:
                data_edge_index_aug = drop_edges(data.edge_index)
                data_aug = Data(edge_index=data_edge_index_aug)
                data_aug_adj = SparseTensor(
                    row=data_edge_index_aug[0],
                    col=data_edge_index_aug[1],
                    sparse_sizes=(self.num_nodes, self.num_nodes),
                )
                data_aug.adj = data_aug_adj
                data_aug.adj_t = data_aug_adj.t()
            else:
                # hypergraph
                data_edge_index_aug = drop_edges(data.edge_index)
                data_aug = Data(edge_index=data_edge_index_aug)
                data_aug.adj_t = data_edge_index_aug

            data_aug = data_aug.to(self.device)

        pos_train_edge, neg_train_edge = get_pos_neg_edges("train", split_edge)

        if "weight" in split_edge["train"]:
            edge_weight_margin = split_edge["train"]["weight"].to(self.device)
        else:
            edge_weight_margin = None

        total_loss = total_examples = 0

        for perm in DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=True):
            self.optimizer.zero_grad()

            input_feat = self.create_input_feat(data)
            if use_drop_edge:
                if data_aug.edge_attr is not None and "MLP" not in self.encoder_name:
                    h = self.encoder(
                        input_feat, data_aug.adj_t, data_aug.edge_attr)
                else:
                    h = self.encoder(input_feat, data_aug.adj_t)
            else:
                if data.edge_attr is not None and "MLP" not in self.encoder_name:
                    h = self.encoder(
                        input_feat, data.adj_t, data.edge_attr
                    )  # input_feat & h both have shape: [N_nodes_full, dim=256]
                else:
                    h = self.encoder(input_feat, data.adj_t)
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
                    self.encoder.parameters(), self.clip_norm
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
    def test(self, data, split_edge, batch_size, evaluator, eval_metric):
        self.encoder.eval()
        self.predictor.eval()

        input_feat = self.create_input_feat(data)
        h = self.encoder(input_feat, data.adj_t)

        # validation set
        pos_val_edge, neg_val_edge = get_pos_neg_edges("val", split_edge)
        pos_val_pred = self.batch_predict(h, pos_val_edge, batch_size)
        neg_val_pred = self.batch_predict(h, neg_val_edge, batch_size)

        # testing set
        pos_head_edge, neg_head_edge = get_pos_neg_edges("head", split_edge)
        pos_tail1_edge, neg_tail1_edge = get_pos_neg_edges("tail1", split_edge)
        pos_tail2_edge, neg_tail2_edge = get_pos_neg_edges("tail2", split_edge)
        pos_iso_edge, neg_iso_edge = get_pos_neg_edges("iso", split_edge)
        pos_head_edge, neg_head_edge = pos_head_edge.to(self.device), neg_head_edge.to(
            self.device
        )
        pos_tail1_edge, neg_tail1_edge = pos_tail1_edge.to(
            self.device
        ), neg_tail1_edge.to(self.device)
        pos_tail2_edge, neg_tail2_edge = pos_tail2_edge.to(
            self.device
        ), neg_tail2_edge.to(self.device)
        pos_iso_edge, neg_iso_edge = pos_iso_edge.to(self.device), neg_iso_edge.to(
            self.device
        )

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
