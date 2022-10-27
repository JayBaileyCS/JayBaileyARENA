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
