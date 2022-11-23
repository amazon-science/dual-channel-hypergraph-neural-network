#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import argparse
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch_geometric.transforms as T
from dual_model import DualModel
from model import BaseModel
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from torch_sparse import SparseTensor
from utils import *


def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--encoder", type=str, default="dual")
    parser.add_argument("--predictor", type=str, default="MLP")
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--loss_func", type=str, default="CL")
    parser.add_argument("--data_name", type=str, default="ogb-citation")
    parser.add_argument("--data_path", type=str, default="dataset")
    parser.add_argument("--walk_start_type", type=str, default="edge")
    parser.add_argument("--res_dir", type=str, default="")
    parser.add_argument("--pretrain_emb", type=str, default="")
    parser.add_argument("--gnn_num_layers", type=int, default=2)
    parser.add_argument("--mlp_num_layers", type=int, default=2)
    parser.add_argument("--emb_hidden_channels", type=int, default=64)
    parser.add_argument("--gnn_hidden_channels", type=int, default=64)
    parser.add_argument("--mlp_hidden_channels", type=int, default=64)
    parser.add_argument("--projection_hidden_channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--grad_clip_norm", type=float, default=2.0)
    parser.add_argument("--batch_size", type=int, default=8 * 1024)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_neg", type=int, default=5)
    parser.add_argument("--walk_length", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--log_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=5)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--year", type=int, default=-1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--lr_min_ratio", type=int, default=0.0001)
    parser.add_argument("--use_lr_decay", type=str2bool, default=False)
    parser.add_argument("--use_node_feats", type=str2bool, default=False)
    parser.add_argument("--use_coalesce", type=str2bool, default=False)
    parser.add_argument("--train_node_emb", type=str2bool, default=True)
    parser.add_argument("--eval_last_best", type=str2bool, default=False)
    args = parser.parse_args()
    return args


def main():
    args = argument()
    set_seed(seed=args.random_seed)
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    data_path = f'./{args.data_path}/{args.data_name}'

    if "citation" in args.data_name:
        paper_paper_graph, paper_paper_hypergraph, split_edge = load_saved_citation_graph(
            data_path, num_neg=args.num_neg)
        if "dual" in args.encoder:
            data, data2 = paper_paper_graph, paper_paper_hypergraph
        elif "hyper" in args.encoder:
            data = paper_paper_hypergraph
        else:
            data = paper_paper_graph
    else:
        author_author_graph, author_author_hypergraph, split_edge = load_saved_collaboration_graph(
            data_path, num_neg=args.num_neg)
        if "dual" in args.encoder:
            data, data2 = author_author_graph, author_author_hypergraph
        elif "hyper" in args.encoder:
            data = author_author_hypergraph
        else:
            data = author_author_graph

    if hasattr(data, "edge_weight"):
        if data.edge_weight is not None:
            data.edge_weight = data.edge_weight.view(-1).to(torch.float)

    if hasattr(data, "num_features"):
        num_node_feats = data.num_features
    else:
        num_node_feats = 0

    if hasattr(data, "num_nodes"):
        num_nodes = data.num_nodes
    else:
        num_nodes = data.adj_t.size(0)

    print(args)

    if hasattr(data, "x"):
        if data.x is not None:
            data.x = data.x.to(torch.float)

    # data = data.to(device)
    if "dual" in args.encoder:
        data2 = data2.to(device)

    if args.encoder.upper() == "GCN":
        # Pre-compute GCN normalization.
        data.adj_t = gcn_normalization(data.adj_t)

    if args.encoder.upper() == "WSAGE":
        data.adj_t = adj_normalization(data.adj_t)

    if args.encoder.upper() == "TRANSFORMER":
        row, col, edge_weight = data.adj_t.coo()
        data.adj_t = SparseTensor(row=row, col=col)

    if "dual" not in args.encoder:
        model = BaseModel(
            lr=args.lr,
            dropout=args.dropout,
            grad_clip_norm=args.grad_clip_norm,
            gnn_num_layers=args.gnn_num_layers,
            mlp_num_layers=args.mlp_num_layers,
            emb_hidden_channels=args.emb_hidden_channels,
            gnn_hidden_channels=args.gnn_hidden_channels,
            mlp_hidden_channels=args.mlp_hidden_channels,
            projection_hidden_channels=args.projection_hidden_channels,
            num_nodes=num_nodes,
            num_node_feats=num_node_feats,
            gnn_encoder_name=args.encoder,
            predictor_name=args.predictor,
            loss_func=args.loss_func,
            optimizer_name=args.optimizer,
            device=device,
            use_node_feats=args.use_node_feats,
            train_node_emb=args.train_node_emb,
            pretrain_emb=args.pretrain_emb,
        )
    else:
        model = DualModel(
            lr=args.lr,
            dropout=args.dropout,
            grad_clip_norm=args.grad_clip_norm,
            gnn_num_layers=args.gnn_num_layers,
            mlp_num_layers=args.mlp_num_layers,
            emb_hidden_channels=args.emb_hidden_channels,
            gnn_hidden_channels=args.gnn_hidden_channels,
            mlp_hidden_channels=args.mlp_hidden_channels,
            projection_hidden_channels=args.projection_hidden_channels,
            num_nodes=num_nodes,
            num_node_feats=num_node_feats,
            gnn_encoder_name1="SAGE",
            gnn_encoder_name2="HYPERGCN",
            predictor_name=args.predictor,
            loss_func=args.loss_func,
            optimizer_name=args.optimizer,
            device=device,
            use_node_feats=args.use_node_feats,
            train_node_emb=args.train_node_emb,
            pretrain_emb=args.pretrain_emb,
        )

    total_params = sum(p.numel() for param in model.para_list for p in param)
    total_params_print = f"Total number of model parameters is {total_params}"
    print(total_params_print)

    print("---- Start Contrastive Learning ----")

    for run in range(args.runs):
        model.param_init()
        start_time = time.time()

        cur_lr = args.lr
        # Contrastive Pretraining
        for epoch in range(1, 1 + args.epochs):
            if "dual" not in args.encoder:
                loss = model.train_cl(data, split_edge, args.batch_size)
            else:
                loss = model.train_cl(data, data2, split_edge, args.batch_size)
            spent_time = time.time() - start_time
            to_print = (
                f"Run: {run + 1:02d}, "
                f"Epoch: {epoch:02d}, "
                f"Loss: {loss:.4f}, "
                f"Learning Rate: {cur_lr:.4f}"
            )
            print(to_print)

            print("---")
            print(f"Training Time Per Epoch: {spent_time: .4f} s")
            print("---")
            start_time = time.time()

            save_dict = {}
            save_dict["state_dict"] = model.state_dict()
            save_path = f"./checkpoint/{args.data_name}_{args.encoder}"
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            torch.save(
                save_dict, f"{save_path}/{args.encoder}_epoch={epoch}.pkl")

            if args.use_lr_decay:
                cur_lr = adjust_lr(model.optimizer, epoch /
                                   args.epochs, args.lr)


if __name__ == "__main__":
    main()
