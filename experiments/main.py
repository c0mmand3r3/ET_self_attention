import math
import torch
from torch import softmax
from torchnlp.word_to_vector import GloVe


def sin_caclulation(pos, i, dmodel):
    return math.sin(pos / math.pow(10000, 2 * i / dmodel))


def cos_caclulation(pos, i, dmodel):
    return math.cos(pos / math.pow(10000, 2 * i / dmodel))


def tranform_vector(vector, tran_vector):
    return vector @ tran_vector.T

sentence = [['name', 'alice', 'cow', 'fast', 'tiger', 'eat', 'kill']]
w_key = [1, 0, 1, 0, 1, 0]
w_query = [1, 2, 3, 0, 1, 0]
w_value = [1, 0, 0, 0, 1, 3]

w_key = torch.tensor(w_key, dtype=torch.float32)
w_query = torch.tensor(w_query, dtype=torch.float32)
w_value = torch.tensor(w_value, dtype=torch.float32)
x = [1.1, 2.2, 1, 3, 4, 1.8]
x_ = torch.tensor(x, dtype=torch.float32)

embedding = [1, 2, 3, 1, 2, 3]

sin_embedding = []
cos_embedding = []
for index, value in enumerate(embedding):
    sin_embedding.append(sin_caclulation(1, 2 * index, value))
    cos_embedding.append(cos_caclulation(1, 2 * index, value))


print(sin_embedding)
print(cos_embedding)
Q = tranform_vector(x_, w_query)
K = tranform_vector(x_, w_key)
T = tranform_vector(x_, w_value)

print(Q)
print(K)
print(T)
attn_scores_softmax = softmax((Q*K.T), dim=-1)
print(attn_scores_softmax)

# x = torch.tensor(x, dtype=torch.float32)
# w_key = [
#   [0, 0, 1],
#   [1, 1, 0],
#   [0, 1, 0],
#   [1, 1, 0]
# ]
# w_query = [
#   [1, 0, 1],
#   [1, 0, 0],
#   [0, 0, 1],
#   [0, 1, 1]
# ]
# w_value = [
#   [0, 2, 0],
#   [0, 3, 0],
#   [1, 0, 3],
#   [1, 1, 0]
# ]
# w_key = torch.tensor(w_key, dtype=torch.float32)
# w_query = torch.tensor(w_query, dtype=torch.float32)
# w_value = torch.tensor(w_value, dtype=torch.float32)
#
# keys = x @ w_key
# querys = x @ w_query
# values = x @ w_value
#
# print(keys)
# # tensor([[0., 1., 1.],
# #         [4., 4., 0.],
# #         [2., 3., 1.]])
#
# print(querys)
# # tensor([[1., 0., 2.],
# #         [2., 2., 2.],
# #         [2., 1., 3.]])
#
# print(values)
# # tensor([[1., 2., 3.],
# #         [2., 8., 0.],
# #         [2., 6., 3.]])
#
# attn_scores = querys @ keys.T
#
# # tensor([[ 2.,  4.,  4.],  # attention scores from Query 1
# #         [ 4., 16., 12.],  # attention scores from Query 2
# #         [ 4., 12., 10.]]) # attention scores from Query 3
#
#
# from torch.nn.functional import softmax
#
# attn_scores_softmax = softmax(attn_scores, dim=-1)
# # tensor([[6.3379e-02, 4.6831e-01, 4.6831e-01],
# #         [6.0337e-06, 9.8201e-01, 1.7986e-02],
# #         [2.9539e-04, 8.8054e-01, 1.1917e-01]])
#
# # For readability, approximate the above as follows
# attn_scores_softmax = [
#   [0.0, 0.5, 0.5],
#   [0.0, 1.0, 0.0],
#   [0.0, 0.9, 0.1]
# ]
# attn_scores_softmax = torch.tensor(attn_scores_softmax)
#
# print(values[:, None])
# exit(0)
#
# weighted_values = values[:,None] * attn_scores_softmax.T[:,:,None]
#
# # tensor([[[0.0000, 0.0000, 0.0000],
# #          [0.0000, 0.0000, 0.0000],
# #          [0.0000, 0.0000, 0.0000]],
# #
# #         [[1.0000, 4.0000, 0.0000],
# #          [2.0000, 8.0000, 0.0000],
# #          [1.8000, 7.2000, 0.0000]],
# #
# #         [[1.0000, 3.0000, 1.5000],
# #          [0.0000, 0.0000, 0.0000],
# #          [0.2000, 0.6000, 0.3000]]])
#
#
# outputs = weighted_values.sum(dim=0)
#
# # tensor([[2.0000, 7.0000, 1.5000],  # Output 1
# #         [2.0000, 8.0000, 0.0000],  # Output 2
# #         [2.0000, 7.8000, 0.3000]]) # Output 3
