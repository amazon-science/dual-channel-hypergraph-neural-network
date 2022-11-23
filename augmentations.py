#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch_sparse import SparseTensor


# graph augmentation, in this repo, we only implement the drop edge (dropE) augmentation.


class Augmentor:
    def __init__(self, aug_ratio=0.4):
        # aug_ratio is to control the level of augmentation, it should be less than 1 and 0 means no augmentation.
        self.aug_ratio = aug_ratio

    def apply_aug(self, edge_index, aug_type="dropE"):
        if aug_type == "dropE":
            aug_edge_index1 = drop_edges(edge_index, self.aug_ratio)
            aug_edge_index2 = drop_edges(edge_index, self.aug_ratio)
        else:
            raise NotImplementedError(
                f"augmentation type {aug_type} not implemented")

        return (aug_edge_index1, aug_edge_index2)


# modify the input parameter aug_ratio to set the augmentation ratio.


def drop_edges(edge_index, aug_ratio=0.4):
    # This function will randomly drop some edges of the original graph to generate a variant augmented graph.
    num_edges = len(edge_index[0])
    drop_num = int(num_edges * aug_ratio)

    idx_perm = np.random.permutation(num_edges)
    edge_idx1 = edge_index[0][idx_perm]
    edge_idx2 = edge_index[1][idx_perm]

    edges_keep1 = edge_idx1[drop_num:]
    edges_keep2 = edge_idx2[drop_num:]

    aug_edge_index = torch.stack([edges_keep1, edges_keep2], dim=0)

    return aug_edge_index
