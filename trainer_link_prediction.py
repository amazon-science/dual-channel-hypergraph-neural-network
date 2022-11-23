#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import argparse
import time
import torch
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
from torch_sparse import coalesce, SparseTensor
from torch_cluster import random_walk
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from sklearn.manifold import TSNE
from model import BaseModel
from dual_model import DualModel
from utils import *


def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--encoder", type=str, default="dual")
    parser.add_argument("--predictor", type=str, default="MLP")
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--loss_func", type=str, default="CE")
    parser.add_argument("--data_name", type=str, default="ogb-citation")
    parser.add_argument("--data_path", type=str, default="dataset")
    parser.add_argument("--eval_metric", type=str, default="recall_my@1")
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
    parser.add_argument("--batch_size", type=int, default=64 * 1024)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_neg", type=int, default=5)
    parser.add_argument("--walk_length", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--log_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=5)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--year", type=int, default=-1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--lr_min_ratio", type=int, default=0.0001)
    parser.add_argument("--use_lr_decay", type=str2bool, default=False)
    parser.add_argument("--use_node_feats", type=str2bool, default=False)
    parser.add_argument("--use_drop_edge", type=str2bool, default=False)
    parser.add_argument("--use_coalesce", type=str2bool, default=False)
    parser.add_argument("--train_node_emb", type=str2bool, default=True)
    parser.add_argument("--eval_last_best", type=str2bool, default=False)
    parser.add_argument("--load", type=str, default=None)
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

    data = data.to(device)
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

    if args.eval_metric == "hits":
        evaluator = Evaluator(name="ogbl-collab")
    elif args.eval_metric == "mrr":
        evaluator = Evaluator(name="ogbl-citation2")
    else:
        evaluator = None

    head_gnn, tail1_gnn, tail2_gnn, iso_gnn = [], [], [], []

    for run in range(args.runs):
        model.param_init()
        start_time = time.time()

        best_val, best_head, best_tail1, best_tail2, best_iso = 0, 0, 0, 0, 0
        best_model_weight = None
        patience_count = 0

        # load pretrained model
        if args.load:
            print("---- Loading Model ----")
            load_dict = torch.load(
                args.load, map_location=f"cuda:{args.device}")
            model.load_state_dict(load_dict["state_dict"])
            print("---- Loading Finished ----")

        cur_lr = args.lr
        # Teacher Training
        for epoch in range(1, 1 + args.epochs):
            if "dual" in args.encoder:
                loss, h = model.train(
                    data,
                    data2,
                    split_edge,
                    batch_size=args.batch_size,
                    num_neg=args.num_neg,
                    use_drop_edge=args.use_drop_edge,
                )
            else:
                loss, h = model.train(
                    data,
                    split_edge,
                    batch_size=args.batch_size,
                    num_neg=args.num_neg,
                    use_drop_edge=args.use_drop_edge,
                )

            if epoch % args.eval_steps == 0:
                if "dual" in args.encoder:
                    results, _ = model.test(
                        data,
                        data2,
                        split_edge,
                        batch_size=args.batch_size,
                        evaluator=evaluator,
                        eval_metric=args.eval_metric,
                    )
                else:
                    results, _ = model.test(
                        data,
                        split_edge,
                        batch_size=args.batch_size,
                        evaluator=evaluator,
                        eval_metric=args.eval_metric,
                    )

                if epoch % args.log_steps == 0:
                    spent_time = time.time() - start_time
                    for key, result in results.items():
                        val_res, head_res, tail1_res, tail2_res, iso_res = result
                        to_print = (
                            f"Run: {run + 1:02d}, "
                            f"Epoch: {epoch:02d}, "
                            f"Loss: {loss:.4f}, "
                            f"Learning Rate: {cur_lr:.4f}, "
                            f"Val: {100 * val_res:.2f}, "
                            f"Head: {100 * head_res:.2f}, "
                            f"Tail1: {100 * tail1_res:.2f}, "
                            f"Tail2: {100 * tail2_res:.2f}, "
                            f"Iso: {100 * iso_res:.2f}"
                        )
                        print(key)
                        print(to_print)

                        if val_res > best_val:
                            patience_count = 0
                            best_val = val_res
                            best_head, best_tail1, best_tail2, best_iso = (
                                head_res,
                                tail1_res,
                                tail2_res,
                                iso_res,
                            )
                            best_model_weight = model.state_dict()
                        else:
                            patience_count += 1

                    print("---")
                    print(
                        f"Training Time Per Epoch: {spent_time / args.eval_steps: .4f} s"
                    )
                    print("---")
                    start_time = time.time()

            if args.use_lr_decay:
                cur_lr = adjust_lr(model.optimizer, epoch /
                                   args.epochs, args.lr)

            if patience_count >= 10:
                print("Teacher GNN Training Finished")
                break

        head_gnn.append(best_head)
        tail1_gnn.append(best_tail1)
        tail2_gnn.append(best_tail2)
        iso_gnn.append(best_iso)

    to_print = (
        f"GNN Results, "
        f"Head: {100 * np.mean(head_gnn):.2f} ± {100 * np.std(head_gnn):.2f}, "
        f"Tail1: {100 * np.mean(tail1_gnn):.2f} ± {100 * np.std(tail1_gnn):.2f} "
        f"Tail2: {100 * np.mean(tail2_gnn):.2f} ± {100 * np.std(tail2_gnn):.2f} "
        f"Iso: {100 * np.mean(iso_gnn):.2f} ± {100 * np.std(iso_gnn):.2f}"
    )

    print(to_print)


if __name__ == "__main__":
    main()
