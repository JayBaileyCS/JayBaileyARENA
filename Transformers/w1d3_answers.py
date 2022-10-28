import torch as t
import einops
from fancy_einsum import einsum


def multihead_masked_attention(Q: t.Tensor, K: t.Tensor, V: t.Tensor, num_heads: int):
    """
    Implements multihead masked attention on the matrices Q, K and V.

    Q: shape (batch, nheads, seq, headsize)
    K: shape (batch, nheads, seq, headsize)
    V: shape (batch, nheads, seq, headsize)
    """
    attention = einsum('batch heads seqQ hidden,batch heads seqK hidden->batch heads seqQ seqK', Q, K)
    attention = attention + t.triu(t.ones_like(attention) * float("-inf"), diagonal=1)
    probs = t.softmax(attention, dim=-1)
    weighted_v = einsum('batch heads seqK hidden, batch heads seqQ seqK -> batch heads seqQ hidden', V, probs)
    return weighted_v


class MultiheadMaskedAttention(t.nn.Module):
    W_QKV: t.nn.Linear
    W_O: t.nn.Linear

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.W_QKV = t.nn.Linear(self.hidden_size, self.num_heads * self.head_size * 3)
        self.W_O = t.nn.Linear(self.num_heads * self.head_size, self.hidden_size)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (batch, seq, hidden_size)

        Return: shape (batch, seq, hidden_size)
        """
        batch, seq, hidden_size = x.shape[0], x.shape[1], x.shape[2]
        QKV = self.W_QKV(x)
        QKV = einops.rearrange(QKV, 'b s (h triplehs) -> b h s triplehs', h=self.num_heads)
        Q = QKV[:, :, :, :self.head_size]
        K = QKV[:, :, :, self.head_size:self.head_size*2]
        V = QKV[:, :, :, self.head_size*2:]
        assert Q.shape == K.shape == V.shape == t.Size([batch, self.num_heads, seq, self.head_size])
        attention = multihead_masked_attention(Q, K, V, self.num_heads)
        assert attention.shape == t.Size([batch, self.num_heads, seq, self.head_size])
        attention = einops.rearrange(attention, 'b n s h -> b s (n h)')
        return self.W_O(attention)
        # return einsum('batch heads seqQ headsize, batch seqQ hidden -> batch seqQ hidden', attention, output)


t.manual_seed(420)
m = MultiheadMaskedAttention(6, 2)
x = t.linspace(0, 42, 2 * 3 * 6).reshape(2, 3, 6)
print(m(x))
