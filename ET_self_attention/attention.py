import math

import torch
import torch.nn as nn

from ET_self_attention.pruning import tile_pruning, row_pruning_with_zeros, scale_and_multiply


class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        WO = torch.randn(size=q.shape, dtype=torch.float32)
        bias = torch.randn(size=q.shape, dtype=torch.float32)

        WQ_ = tile_pruning(q, bias, 2, 2, milesone=True).transpose(-2, -1)
        WK_ = tile_pruning(k, bias, 2, 2, milesone=True).transpose(-2, -1)
        WV_ = row_pruning_with_zeros(v, 2).transpose(-2, -1)

        x_ = torch.reshape(x, q.shape)

        q_ = torch.matmul(x_.float(), WQ_.float())
        k_ = torch.matmul(x_.float(), WK_.float())
        v_ = torch.matmul(x_.float(), WV_.float())

        k_transpose = k_.transpose(-2, -1)
        d_k = q.size()[-1]

        scale_values = scale_and_multiply(q_, d_k, k_transpose)

        z = torch.matmul(scale_values, v_)
        z[z != z] = 0
        WO = tile_pruning(WO, bias, 2, 2, milesone=True)

        output = torch.matmul(z.float(), WO.float())
        # add_val = torch.zeros(x_.shape, dtype=torch.float32)
        # for val in range(output.shape[0]):
        #     add_val += output[val]
        add_val = torch.reshape(output, x.shape)
        return add_val




class MultiheadAttention_secondary(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        WO = torch.randn(size=q.shape, dtype=torch.float32)
        bias = torch.randn(size=q.shape, dtype=torch.float32)

        WQ_ = tile_pruning(q, bias, 2, 2, milesone=True).transpose(-2, -1)
        WK_ = tile_pruning(k, bias, 2, 2, milesone=True).transpose(-2, -1)
        WV_ = v.transpose(-2, -1)
        WO_ = row_pruning_with_zeros(WO, 2)

        x_ = torch.reshape(x, torch.Size((x.shape[0], 1, x.shape[1], x.shape[2])))

        q_ = torch.matmul(x_, WQ_)
        k_ = torch.matmul(x_, WK_)

        precompute = torch.matmul(WV_.transpose(-2, -1), WO_.transpose(-2, -1))

        k_transpose = k_.transpose(-2, -1)
        d_k = q.size()[-1]

        scale_values_ = scale_and_multiply(q_, d_k, k_transpose)


        preoutput = torch.matmul(precompute, x_)
        output = torch.matmul(scale_values_, preoutput)
        # scale_values = scale_values_/ math.sqrt(d_k)
        # z = torch.matmul(scale_values, v)
        # z[z != z] = 0
        # WO = tile_pruning(WO, 2, 2)
        # # WO_ = WO.transpose(-2, -1)
        # print(z.shape)
        # print(WO.shape)
        # output = torch.matmul(z, WO)
        add_val = torch.reshape(output, x.shape)
        return add_val
