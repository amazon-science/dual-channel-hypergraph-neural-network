#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import os
import random
import time
import torch
from collections import OrderedDict
from torch_geometric.utils import *
from torch_geometric.loader import *
from torch_geometric.data import *
from torch_geometric.datasets import *
from torch_sparse import SparseTensor


def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default="./dataset")
    parser.add_argument("--sampled_authors", type=int, default=50000)
    args = parser.parse_args()
    return args


def filter(edge_index, hyperedge_index):
    # This function will relabel nodes and hyperedge index starting from 0
    print("---- filter start ----")
    node2id = {}
    start = 0
    # construct the convert table for old node id to new node id
    for nodes in edge_index:
        for node in nodes:
            if int(node) not in node2id:
                node2id[int(node)] = start
                start += 1

    edge_index.apply_(lambda x: node2id[int(x)])

    filterred_hyperedge_index = torch.LongTensor()
    for i in range(len(hyperedge_index[0])):
        if int(hyperedge_index[0][i]) in node2id:
            filterred_hyperedge_index = torch.cat(
                (filterred_hyperedge_index, hyperedge_index[:, i].view((2, -1))), dim=1
            )

    hyperedge2id = {}
    start = 0
    # construct the convert table for old hyperedge index to new hyperedge index
    for hyperedge in hyperedge_index[1]:
        if int(hyperedge) not in hyperedge2id:
            hyperedge2id[int(hyperedge)] = start
            start += 1

    filterred_hyperedge_index[0].apply_(lambda x: node2id[int(x)])
    filterred_hyperedge_index[1].apply_(lambda x: hyperedge2id[int(x)])
    print("---- filter done ----")

    return edge_index, filterred_hyperedge_index


def main():
    args = argument()

    # get the raw dataset (ogb_mag)
    dataset = OGB_MAG(root="./data", preprocess="metapath2vec")
    data = dataset[0]

    print("---- citation dataset preprocessing ----")

    # citation networks preprocess
    # citation normal graph edge index and topic hyperedge index
    cites_edge_data = data["cites"]["edge_index"]
    topic_edge_data = data["has_topic"]["edge_index"]

    # create a data folder to store the processed citation data
    citation_output_path = f"{args.save_path}/ogb-citation"
    if not os.path.exists(citation_output_path):
        os.makedirs(citation_output_path)

    np.save(
        f"{citation_output_path}/edge_index.npy",
        cites_edge_data.numpy(),
    )
    np.save(
        f"{citation_output_path}/hyperedge_index.npy",
        topic_edge_data.numpy(),
    )

    print("---- citation dataset preprocessed ----")

    print("---- collaboration dataset preprocessing ----")

    # collaboration networks preprocess
    # collaboration normal graph edge index and institution hyperedge index
    writes_edge_data = data["writes"]["edge_index"]
    affiliated_edge_data = data["affiliated_with"]["edge_index"]

    author_node_num = data["author"].x.size(0)
    institution_node_num = data["institution"].x.size(0)

    # relabel the node
    source, target = writes_edge_data[0], writes_edge_data[1]
    new_target = target + author_node_num

    author_to_paper = torch.stack([source, new_target], dim=0)
    edge_index = to_undirected(author_to_paper, reduce="add")

    adj = SparseTensor(row=edge_index[0], col=edge_index[1])

    # put it to GPU to speed the process, CPU is also okay.
    # the following process is to get the author-author co-authorship graph
    adj = adj.to(device=torch.device("cuda:0"))
    adj = adj.matmul(adj)
    adj = adj.cpu()[:author_node_num, :author_node_num]
    row, col, value = adj.coo()
    edge_index = torch.stack([row, col], dim=0)
    edge_index, _ = remove_self_loops(edge_index)

    # filter the authors who have the information of institution information
    unique_authors = torch.unique(edge_index, sorted=True)
    # authors who have institution information
    existing_authors = torch.unique(affiliated_edge_data[0], sorted=True)
    overlap_authors = np.intersect1d(unique_authors, existing_authors)

    # get subgraphs of the overlap authors (sampled), the whole graph is too large to be loaded in a single GPU
    edge_index, _ = subgraph(torch.tensor(
        overlap_authors[:args.sampled_authors]), edge_index)
    hyperedge_index, _ = subgraph(
        torch.tensor(
            overlap_authors[:args.sampled_authors]), affiliated_edge_data
    )

    edge_index, hyperedge_index = filter(edge_index, hyperedge_index)

    # get the missing authors who do not have institution information and set the virtual information for them
    unique_authors_in_edge_index = torch.unique(edge_index, sorted=True)
    unique_authors_in_hyperedge_index = torch.unique(
        hyperedge_index[0], sorted=True)

    missing_authors = np.setdiff1d(
        unique_authors_in_edge_index, unique_authors_in_hyperedge_index
    )
    virtual_institution_index = np.arange(
        start=hyperedge_index[1].max(),
        stop=hyperedge_index[1].max() + len(missing_authors),
    )

    virtual_affiliated_edge_data = torch.stack(
        [torch.tensor(missing_authors), torch.tensor(virtual_institution_index)], dim=0
    )
    hyperedge_index = torch.concat(
        [hyperedge_index, virtual_affiliated_edge_data], dim=1
    )

    # create a data folder to store the processed collaboration data
    collaboration_output_path = f"{args.save_path}/ogb-collaboration"
    if not os.path.exists(collaboration_output_path):
        os.makedirs(collaboration_output_path)

    np.save(
        f"{collaboration_output_path}/edge_index.npy",
        edge_index.numpy(),
    )
    np.save(
        f"{collaboration_output_path}/hyperedge_index.npy",
        hyperedge_index.numpy(),
    )

    print("---- collaboration dataset preprocessed ----")


if __name__ == "__main__":
    main()
