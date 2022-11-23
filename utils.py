#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  SPDX-License-Identifier: Apache-2.0
import argparse
import torch
import os
import numpy as np
import random
import time
from collections import defaultdict, Counter, OrderedDict
from functools import reduce
from torch_geometric.utils import *
from torch_geometric.loader import *
from torch_geometric.data import *
from torch_sparse import SparseTensor


def set_seed(seed):
    """
    Making training deterministic.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def adjust_lr(optimizer, decay_ratio, min_ratio, lr):
    lr_ = lr * (1 - decay_ratio)
    lr_min = lr * min_ratio
    if lr_ < lr_min:
        lr_ = lr_min
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr_
    return lr_


def load_saved_citation_graph(data_path, num_neg=50):
    print(f"---- loading dataset ----")
    sampled_edge_index = torch.from_numpy(
        np.load(f"{data_path}/edge_index.npy"))
    sampled_hyperedge_index = torch.from_numpy(
        np.load(f"{data_path}/hyperedge_index.npy"))

    print(f"---- spliting edges ----")
    split_edge = edge_split_ogb(
        sampled_edge_index, sampled_hyperedge_index, num_neg)
    print(f"---- edges split done ----")

    paper_paper_graph = Data(edge_index=sampled_edge_index)
    paper_paper_adj = SparseTensor(
        row=sampled_edge_index[0], col=sampled_edge_index[1])
    paper_paper_graph.adj = paper_paper_adj
    paper_paper_graph.adj_t = paper_paper_adj.t()
    paper_paper_graph.num_nodes = paper_paper_adj.size(0)

    paper_paper_hypergraph = Data(edge_index=sampled_hyperedge_index)
    paper_paper_hypergraph.adj_t = sampled_hyperedge_index
    paper_paper_hypergraph.num_nodes = paper_paper_adj.size(0)

    return paper_paper_graph, paper_paper_hypergraph, split_edge


def load_saved_collaboration_graph(data_path, num_neg=50):
    print(f"---- loading dataset ----")
    sampled_edge_index = torch.from_numpy(
        np.load(f"{data_path}/edge_index.npy"))
    sampled_hyperedge_index = torch.from_numpy(
        np.load(f"{data_path}/hyperedge_index.npy"))

    print(f"---- spliting edges ----")
    split_edge = edge_split_ogb(sampled_edge_index, sampled_hyperedge_index, num_neg)
    print(f"---- edges split done ----")
    # create undirected author-author graph
    edge_index = to_undirected(sampled_edge_index, reduce="add")

    author_author_graph = Data(edge_index=edge_index)
    author_author_adj = SparseTensor(row=edge_index[0], col=edge_index[1])
    author_author_graph.adj = author_author_adj
    author_author_graph.adj_t = author_author_adj.t()
    author_author_graph.num_nodes = author_author_adj.size(0)

    author_author_hypergraph = Data(edge_index=sampled_hyperedge_index)
    author_author_hypergraph.adj_t = sampled_hyperedge_index
    author_author_hypergraph.num_nodes = author_author_adj.size(0)

    return author_author_graph, author_author_hypergraph, split_edge


# For mrr evaluation
def get_pos_neg_edges(split, split_edge):
    pos_edge = split_edge[split]["edge"]
    neg_edge = split_edge[split]["edge_neg"]
    return pos_edge, neg_edge


def tonp(arr):
    if type(arr) is torch.Tensor:
        return arr.detach().cpu().data.numpy()
    else:
        return np.asarray(arr)


def toitem(arr, round=True):
    arr1 = tonp(arr)
    value = arr1.reshape(-1)[0]
    if round:
        value = np.round(value, 3)
    assert arr1.size == 1
    return value


def evaluate_hits(
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
):
    results = {}
    for K in [20, 50, 100]:
        evaluator.K = K
        val_hits = evaluator.eval(
            {
                "y_pred_pos": pos_val_pred,
                "y_pred_neg": neg_val_pred,
            }
        )[f"hits@{K}"]
        head_hits = evaluator.eval(
            {
                "y_pred_pos": pos_head_pred,
                "y_pred_neg": neg_head_pred,
            }
        )[f"hits@{K}"]
        tail1_hits = evaluator.eval(
            {
                "y_pred_pos": pos_tail1_pred,
                "y_pred_neg": neg_tail1_pred,
            }
        )[f"hits@{K}"]
        tail2_hits = evaluator.eval(
            {
                "y_pred_pos": pos_tail2_pred,
                "y_pred_neg": neg_tail2_pred,
            }
        )[f"hits@{K}"]
        iso_hits = evaluator.eval(
            {
                "y_pred_pos": pos_iso_pred,
                "y_pred_neg": neg_iso_pred,
            }
        )[f"hits@{K}"]

        results[f"Hits@{K}"] = (val_hits, head_hits,
                                tail1_hits, tail2_hits, iso_hits)

    return results


def evaluate_mrr(
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
):
    neg_val_pred = neg_val_pred.view(pos_val_pred.shape[0], -1)
    neg_head_pred = neg_head_pred.view(pos_head_pred.shape[0], -1)
    neg_tail1_pred = neg_tail1_pred.view(pos_tail1_pred.shape[0], -1)
    neg_tail2_pred = neg_tail2_pred.view(pos_tail2_pred.shape[0], -1)
    neg_iso_pred = neg_iso_pred.view(pos_iso_pred.shape[0], -1)
    results = {}
    val_mrr = evaluator.eval({
        'y_pred_pos': pos_val_pred,
        'y_pred_neg': neg_val_pred,
    })['mrr_list'].mean().item()

    head_mrr = evaluator.eval({
        'y_pred_pos': pos_head_pred,
        'y_pred_neg': neg_head_pred,
    })['mrr_list'].mean().item()

    tail1_mrr = evaluator.eval({
        'y_pred_pos': pos_tail1_pred,
        'y_pred_neg': neg_tail1_pred,
    })['mrr_list'].mean().item()

    tail2_mrr = evaluator.eval({
        'y_pred_pos': pos_tail2_pred,
        'y_pred_neg': neg_tail2_pred,
    })['mrr_list'].mean().item()

    iso_mrr = evaluator.eval({
        'y_pred_pos': pos_iso_pred,
        'y_pred_neg': neg_iso_pred,
    })['mrr_list'].mean().item()

    results["MRR"] = (val_mrr, head_mrr, tail1_mrr, tail2_mrr, iso_mrr)

    return results


def cal_recall(pos_score_1D, neg_score_1D, topk=None):
    if topk is None or float(topk) == 0:  # default threshold by 0.
        return toitem((pos_score_1D > 0).sum()) / len(pos_score_1D)
    elif float(topk) > 5:  # absolute value
        topk = int(topk)
    else:  # relative value
        topk = int(float(topk) * len(pos_score_1D))

    N_pos_total = len(pos_score_1D)
    force_greater_0 = 1
    if force_greater_0:
        pos_score_1D = pos_score_1D[pos_score_1D > 0]

    scores = np.concatenate([tonp(pos_score_1D), tonp(neg_score_1D)])
    labels = np.concatenate(
        [np.ones(len(pos_score_1D)), np.zeros(len(neg_score_1D))])
    arr = np.asarray([scores, labels]).T  # shape=[N,2]
    sortarr = np.asarray(
        sorted(list(arr), key=lambda x: x[0], reverse=True))  # [N,2]
    recall = sortarr[:topk, 1].sum() / N_pos_total
    return recall


def evaluate_recall_my(
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
    topk=None,
):
    results = {}
    recall_val = cal_recall(pos_val_pred, neg_val_pred, topk=topk)
    recall_head = cal_recall(pos_head_pred, neg_head_pred, topk=topk)
    recall_tail1 = cal_recall(pos_tail1_pred, neg_tail1_pred, topk=topk)
    recall_tail2 = cal_recall(pos_tail2_pred, neg_tail2_pred, topk=topk)
    recall_iso = cal_recall(pos_iso_pred, neg_iso_pred, topk=topk)
    results["recall@100%"] = (
        recall_val,
        recall_head,
        recall_tail1,
        recall_tail2,
        recall_iso,
    )

    return results


def gcn_normalization(adj_t):
    adj_t = adj_t.set_diag()
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    return adj_t


def adj_normalization(adj_t):
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-1)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t
    return adj_t


# Edge Split For Citation Networks and Collaboration Networks
# generate negative edges for positive edges, for each positive edges, sample 100 negative edges, 50 for head and 50 for tail
def my_ogb_negative_sampling(positive_edge, num_of_src, num_of_dst, num_neg=50):
    negative_edge = []
    for src, dst in positive_edge:
        neg_dst = torch.tensor(random.sample(range(num_of_dst), num_neg))
        for n_dst in neg_dst:
            negative_edge.append([src, n_dst])
        neg_src = torch.tensor(random.sample(range(num_of_src), num_neg))
        for n_src in neg_src:
            negative_edge.append([n_src, dst])

    return torch.tensor(negative_edge)


# split edges into six groups: training, validation, and testing (head, tail1, tail2, and iso).
def edge_split_ogb(edge_index, hyperedge_index, num_neg=5):
    # normal graph adj
    normal_adj = SparseTensor(row=edge_index[0], col=edge_index[1])
    # node degree matrix
    normal_D = normal_adj.sum(dim=1) + normal_adj.sum(0)
    hyper_adj = SparseTensor(row=hyperedge_index[0], col=hyperedge_index[1])
    hyper_edge_D = hyper_adj.sum(dim=0)

    node_degree_bipar, node_degree_hyper = {}, {}

    # get asin node degree via bipartite asin-query graph
    for i, value in enumerate(normal_D):
        node_degree_bipar[i] = int(value)

    # get asin node degree via asin-asin hypergraph
    for i in range(len(hyperedge_index[0])):
        node, hyperedge = hyperedge_index[:, i]
        if int(node) not in node_degree_hyper:
            node_degree_hyper[int(node)] = int(hyper_edge_D[hyperedge])
        else:
            node_degree_hyper[int(node)] += int(hyper_edge_D[hyperedge])

    # sort asin node degree for bipartite graph
    sorted_node_degree_bipar = OrderedDict(
        sorted(node_degree_bipar.items(), key=lambda x: x[1]))
    sorted_node_degree_bipar = list(sorted_node_degree_bipar.items())

    num_of_nodes = len(sorted_node_degree_bipar)
    # num_of_querys = max(asin_query_edge_index[1]) - max(asin_query_edge_index[0])
    num_of_edges = len(edge_index[0])
    partion = num_of_nodes // 5
    node_index = np.arange(num_of_nodes)

    # split asin node into three parts via bipartite graph, head (20%), tail (40%) and iso (40%)
    head_bipar, tail_bipar, iso_bipar = [], [], []
    for i in range(len(sorted_node_degree_bipar)):
        node, _ = sorted_node_degree_bipar[i]
        if i <= 2 * partion:
            iso_bipar.append(node)
        elif 2 * partion < i <= 4 * partion:
            tail_bipar.append(node)
        else:
            head_bipar.append(node)

    # sort asin node degree for hypergraph
    sorted_node_degree_hyper = OrderedDict(
        sorted(node_degree_hyper.items(), key=lambda x: x[1]))
    sorted_node_degree_hyper = list(sorted_node_degree_hyper.items())

    # split asin node into three parts via hyper graph, head (20%), tail (40%) and iso (40%)
    head_hyper, tail_hyper, iso_hyper = [], [], []
    for i in range(len(sorted_node_degree_hyper)):
        node, _ = sorted_node_degree_hyper[i]
        if i <= 2 * partion:
            iso_hyper.append(node)
        elif 2 * partion < i <= 4 * partion:
            tail_hyper.append(node)
        else:
            head_hyper.append(node)

    # generate head, tail1, tail2, and iso set for testing, the remaining nodes are for training
    head = np.intersect1d(head_bipar, head_hyper)
    tail1 = np.intersect1d(head_bipar, tail_hyper)
    tail2 = np.intersect1d(tail_bipar, head_hyper)
    iso = np.intersect1d(iso_bipar, iso_hyper)
    train = np.setdiff1d(node_index, np.concatenate((head, tail1, tail2, iso)))

    # generate positive and negative edges for the five parts
    _, p_edge_train, _, _ = k_hop_subgraph(torch.tensor(
        train), 1, edge_index, flow='target_to_source')
    _, p_edge_head, _, _ = k_hop_subgraph(torch.tensor(
        head), 1, edge_index, flow='target_to_source')
    _, p_edge_tail1, _, _ = k_hop_subgraph(torch.tensor(
        tail1), 1, edge_index, flow='target_to_source')
    _, p_edge_tail2, _, _ = k_hop_subgraph(torch.tensor(
        tail2), 1, edge_index, flow='target_to_source')
    _, p_edge_iso, _, _ = k_hop_subgraph(torch.tensor(
        iso), 1, edge_index, flow='target_to_source')
    p_edge_train, p_edge_head, p_edge_tail1, p_edge_tail2, p_edge_iso = p_edge_train.T, p_edge_head.T, p_edge_tail1.T, p_edge_tail2.T, p_edge_iso.T

    num_of_portion = int(0.05 * num_of_edges)
    head_index = torch.randperm(len(p_edge_head))
    p_edge_head_testing = p_edge_head[head_index[:num_of_portion]]
    p_edge_head_validation = p_edge_head[head_index[num_of_portion:2*num_of_portion]]
    p_edge_head_training = p_edge_head[head_index[2*num_of_portion:]]

    tail1_index = torch.randperm(len(p_edge_tail1))
    p_edge_tail1_testing = p_edge_tail1[tail1_index[:num_of_portion]]
    p_edge_tail1_validation = p_edge_tail1[tail1_index[num_of_portion:2*num_of_portion]]
    p_edge_tail1_training = p_edge_tail1[tail1_index[2*num_of_portion:]]

    tail2_index = torch.randperm(len(p_edge_tail2))
    p_edge_tail2_testing = p_edge_tail2[tail2_index[:num_of_portion]]
    p_edge_tail2_validation = p_edge_tail2[tail2_index[num_of_portion:2*num_of_portion]]
    p_edge_tail2_training = p_edge_tail2[tail2_index[2*num_of_portion:]]

    iso_index = torch.randperm(len(p_edge_iso))
    p_edge_iso_testing = p_edge_iso[iso_index[:num_of_portion]]
    p_edge_iso_validation = p_edge_iso[iso_index[num_of_portion:2*num_of_portion]]
    p_edge_iso_training = p_edge_iso[iso_index[2*num_of_portion:]]

    p_edge_train = torch.cat([p_edge_train, p_edge_head_training,
                             p_edge_tail1_training, p_edge_tail2_training, p_edge_iso_training])
    p_edge_val = torch.cat([p_edge_head_validation, p_edge_tail1_validation,
                           p_edge_tail2_validation, p_edge_iso_validation])

    # generate negative edges for positive edges, for each positive edges, sample 10 negative edges, 5 for head and 5 for tail
    n_edge_train = my_ogb_negative_sampling(
        p_edge_train, num_of_nodes, num_of_nodes)
    n_edge_val = my_ogb_negative_sampling(
        p_edge_val, num_of_nodes, num_of_nodes, num_neg)
    n_edge_head_testing = my_ogb_negative_sampling(
        p_edge_head_testing, num_of_nodes, num_of_nodes, 50)
    n_edge_tail1_testing = my_ogb_negative_sampling(
        p_edge_tail1_testing, num_of_nodes, num_of_nodes, 50)
    n_edge_tail2_testing = my_ogb_negative_sampling(
        p_edge_tail2_testing, num_of_nodes, num_of_nodes, 50)
    n_edge_iso_testing = my_ogb_negative_sampling(
        p_edge_iso_testing, num_of_nodes, num_of_nodes, 50)

    print(
        f'num of train positive edges: {len(p_edge_train)}\nnum of train negative edges: {len(n_edge_train)}')
    print(
        f'num of val positive edges: {len(p_edge_val)}\nnum of val negative edges: {len(n_edge_val)}')
    print(
        f'num of head positive edges: {len(p_edge_head_testing)}\nnum of head negative edges: {len(n_edge_head_testing)}')
    print(
        f'num of tail1 positive edges: {len(p_edge_tail1_testing)}\nnum of tail1 negative edges: {len(n_edge_tail1_testing)}')
    print(
        f'num of tail2 positive edges: {len(p_edge_tail2_testing)}\nnum of tail2 negative edges: {len(n_edge_tail2_testing)}')
    print(
        f'num of iso positive edges: {len(p_edge_iso_testing)}\nnum of iso negative edges: {len(n_edge_iso_testing)}')

    # ---- save split edges ----
    split_edge = {
        'train': {'edge': p_edge_train, 'edge_neg': n_edge_train},
        'val': {'edge': p_edge_val, 'edge_neg': n_edge_val},
        'head': {'edge': p_edge_head_testing, 'edge_neg': n_edge_head_testing},
        'tail1': {'edge': p_edge_tail1_testing, 'edge_neg': n_edge_tail1_testing},
        'tail2': {'edge': p_edge_tail2_testing, 'edge_neg': n_edge_tail2_testing},
        'iso': {'edge': p_edge_iso_testing, 'edge_neg': n_edge_iso_testing}
    }

    return split_edge
