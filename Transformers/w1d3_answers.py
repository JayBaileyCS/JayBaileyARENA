import torch as t
import einops


def multihead_masked_attention(Q: t.Tensor, K: t.Tensor, V: t.Tensor, num_heads: int):
    """
    Implements multihead masked attention on the matrices Q, K and V.

    Q: shape (batch, seq, nheads*headsize)
    K: shape (batch, seq, nheads*headsize)
    V: shape (batch, seq, nheads*headsize)
    """
    Q = einops.rearrange(Q, 'b s (h hs) -> b h s hs', h=num_heads)
    K = einops.rearrange(K, 'b s (h hs) -> b h s hs', h=num_heads)
    V = einops.rearrange(V, 'b s (h hs) -> b h s hs', h=num_heads)
    attention = t.einsum('bhqt,bhkt->bhqk', Q, K)
    attention = attention + t.triu(t.ones_like(attention) * float("-inf"), diagonal=1)
    probs = t.softmax(attention, dim=-1)
    weighted_v = t.einsum('bnkh, bnsk -> bnsh', V, probs)
    weighted_v = einops.rearrange(weighted_v, 'b h s hs -> b s (h hs)')
    return weighted_v  # Output needs to be applied?


class MultiheadMaskedAttention(t.nn.Module):
    W_QKV: t.nn.Linear
    W_O: t.nn.Linear

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size / num_heads

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (batch, seq, hidden_size)

        Return: shape (batch, seq, hidden_size)
        """
        W_Q = self.W_QKV[:, :, :self.hidden_size]
        W_K = self.W_QKV[:, :, self.hidden_size+1:self.hidden_size*2]
        W_V = self.W_QKV[:, :, self.hidden_size*2+1:]
        Q = t.einsum('bij, bjk -> bik', x, W_Q)
        K = t.einsum('bij, bjk -> bik', x, W_K)
        V = t.einsum('bij, bjk -> biko', x, W_V)
        attention = multihead_masked_attention(Q, K, V, self.num_heads)
        return t.einsum('bsh,bho->bso', attention, self.W_O)
