import math
import einops
import torch as t
from w1d1_answers import calculate_positional_encoding


class PositionalEncoding(t.nn.Module):

    def __init__(self, embedding_dim: int, max_seq_len: int = 5000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len

    def forward(self, x: t.Tensor) -> t.Tensor:
        """x: shape (batch, seq_len, embedding_dim)"""
        encoding = t.tensor(calculate_positional_encoding(self.embedding_dim, self.max_seq_len))
        return x + einops.repeat(encoding, 's e -> b s e', b=x.shape[0])


def single_head_attention(Q: t.Tensor, K: t.Tensor, V: t.Tensor) -> t.Tensor:
    """
    Should return the results of self-attention (see the "Self-Attention in Detail" section of the Illustrated Transformer).

    With this function, you can ignore masking.

    Q: shape (batch, seq_length, hidden_size)
    K: shape (batch, seq_length, hidden_size)
    V: shape (batch, seq_length, hidden_size)
    hidden_size = head_size * number_of_heads.

    Return: shape (batch, seq_len, hidden_size)
    """
    attention = t.einsum('bsh, bth -> bst', Q, K)
    attention = t.softmax(attention / math.sqrt(Q.shape[-1]), dim=-1)
    return t.einsum('bst, btv -> bsv', attention, V)  # Unsure about this one.


def single_head_masked_attention(Q: t.Tensor, K: t.Tensor, V: t.Tensor) -> t.Tensor:
    """
    Should return the results of masked self-attention.

    See "The Decoder Side" section of the Illustrated Transformer for an explanation of masking.

    Q: shape (batch, embedding_size, head_size)
    K: shape (batch, embedding_size, head_size)
    V: shape (batch, embedding_size, value_size)

    Return: shape (batch, seq_len, embedding_size)
    """
    attention = t.einsum('bsh, bth -> bst', Q, K)
    attention = attention + t.triu(t.ones_like(attention) * float("-inf"), diagonal=1)  # Add mask
    attention = t.softmax(attention / math.sqrt(Q.shape[-1]), dim=-1)
    return t.einsum('bst, btv -> bsv', attention, V)  # Unsure about this one.


Q = t.arange(2 * 7 * 4).reshape(2, 7, 4).type(t.float32)
K = Q * 0.5
V = Q * 0.8
num_heads = 1
print(single_head_masked_attention(Q, K, V))
