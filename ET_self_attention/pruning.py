import math

import numpy as np
import torch
import torch.nn.functional as F


def scale_and_multiply(q, dk, k_transpose, mask=None):
    scale_q = q / math.sqrt(dk)
    attn_logits = torch.matmul(scale_q, k_transpose)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    return attention



def tile_pruning(matrix, bias, r, c, milesone=False, alpha=0.01, ep=-0.01, percentile=0.5):

    if milesone:
        l2_norm = torch.zeros(size=(matrix.shape[0], matrix.shape[1],
                                    int(matrix.shape[2] / r), int(matrix.shape[3] / c)), dtype=torch.float32)
        penalty_factor = torch.zeros(size=(matrix.shape[0], matrix.shape[1],
                                           int(matrix.shape[2] / r), int(matrix.shape[3] / c)), dtype=torch.float32)
        row_initial = 0
        for row in range(0, matrix.shape[2], r):
            col_initial = 0
            for col in range(0, matrix.shape[3], c):
                sub_matrix = matrix[:, :, row_initial:row + r, col_initial:col + c]
                sub_matrix = sub_matrix.detach().numpy()

                for val in range(sub_matrix.shape[0]):
                    l2_norm[val, :, int(row / r), int(col / c)] = float(np.linalg.norm(sub_matrix[val][0]))
                    penalty_factor[val, :, int(row / r), int(col / c)] = \
                        (1 / (l2_norm[val, :, int(row / r), int(col / c)] + ep))

                col_initial += c
            row_initial += r
        regularization_term = torch.sum(l2_norm * penalty_factor) * alpha
        import torch.nn as nn
        loss = nn.L1Loss()
        output = loss(matrix, bias)
        total_loss = output + regularization_term
        matrix = matrix - total_loss
    maximum = -10
    if matrix.shape[2] % r != 0 or matrix.shape[3] % c != 0:
        return matrix
    row_initial = 0
    for row in range(0, matrix.shape[2], r):
        col_initial = 0
        for col in range(0, matrix.shape[3], c):
            sub_matrix = matrix[:, :, row_initial:row + r, col_initial:col + c]
            sub_matrix = sub_matrix.detach().numpy()

            for val in range(sub_matrix.shape[0]):
                maximum = max(float(np.linalg.norm(sub_matrix[val][0])), maximum)
            col_initial += c
        row_initial += r
    threshold = maximum * percentile
    major_matrix = torch.zeros(matrix.shape, dtype=torch.float32)
    row_initial = 0
    for row in range(0, matrix.shape[2], r):
        col_initial = 0
        for col in range(0, matrix.shape[3], c):
            sub_matrix = matrix[:, :, row_initial:row + r, col_initial:col + c]
            sub_matrix = sub_matrix.detach().numpy()

            sub_initial = 0
            for val in range(sub_matrix.shape[0]):

                value_norm = float(np.linalg.norm(sub_matrix[val][0]))
                if value_norm >= threshold:
                    major_matrix[sub_initial:val + 1, :, row_initial:row + r, col_initial:col + c] = torch.from_numpy(
                        sub_matrix[val][0])
                    sub_initial += 1
                else:
                    sub_initial += 1
            col_initial += c
        row_initial += r

    if matrix.shape == major_matrix.shape:
        return major_matrix
    else:
        return None


def column_pruning(src_matrix, col):
    matrix = torch.zeros(size=(src_matrix.shape[0], src_matrix.shape[1],
                               src_matrix.shape[2], int(src_matrix.shape[3] / col)), dtype=torch.float32)
    for index, prune_col in enumerate(range(0, src_matrix.shape[3], 2)):
        sub_matrix = src_matrix[:, :, :, prune_col]
        matrix[:, :, :, index] = sub_matrix
    return matrix


def row_pruning(src_matrix, row):
    matrix = torch.zeros(size=(src_matrix.shape[0], src_matrix.shape[1],
                               int(src_matrix.shape[2] / row), src_matrix.shape[3]), dtype=torch.float32)
    for index, prune_row in enumerate(range(0, src_matrix.shape[2], 2)):
        sub_matrix = src_matrix[:, :, prune_row, :]
        matrix[:, :, index, :] = sub_matrix
    return matrix


def row_pruning_with_zeros(src_matrix, row):
    matrix = torch.zeros(size=src_matrix.shape, dtype=torch.float32)
    for index, prune_row in enumerate(range(0, src_matrix.shape[2], 2)):
        sub_matrix = src_matrix[:, :, prune_row, :]
        matrix[:, :, prune_row, :] = sub_matrix
    return matrix
