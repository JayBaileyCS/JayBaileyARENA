from dataclasses import dataclass
import torch as t
import einops
from fancy_einsum import einsum
from torch.utils.data import DataLoader, Dataset
import tqdm

from Transformers.w1d1_answers import calculate_positional_encoding


def multihead_masked_attention(Q: t.Tensor, K: t.Tensor, V: t.Tensor):
    """
    Implements multihead masked attention on the matrices Q, K and V.
    Q, K, V: shape (batch, nheads, seq, headsize)
    """
    attention = einsum('batch nheads seqQ headsize, batch nheads seqK headsize -> batch nheads seqQ seqK', Q, K)
    attention = attention + t.triu(t.ones_like(attention) * float("-inf"), diagonal=1)
    probs = t.softmax(attention, dim=-1)
    weighted_v = einsum('batch nheads seqK headsize, batch nheads seqQ seqK -> batch nheads seqQ headsize', V, probs)
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
        batch, seq, hidden_size = x.shape
        QKV = einops.rearrange(self.W_QKV(x), 'b s (n triplehs) -> b n s triplehs', n=self.num_heads)
        Q = QKV[:, :, :, :self.head_size]
        K = QKV[:, :, :, self.head_size:self.head_size * 2]
        V = QKV[:, :, :, self.head_size * 2:]
        assert Q.shape == K.shape == V.shape == t.Size([batch, self.num_heads, seq, self.head_size])
        attention = multihead_masked_attention(Q, K, V)
        assert attention.shape == t.Size([batch, self.num_heads, seq, self.head_size])
        attention = einops.rearrange(attention, 'b n s h -> b s (n h)')
        return self.W_O(attention)


@dataclass(frozen=True)
class TransformerConfig:
    """Constants used throughout your decoder-only transformer model."""
    num_layers: int
    num_heads: int
    vocab_size: int
    hidden_size: int
    max_seq_len: int
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-05


class PositionalEncoding(t.nn.Module):

    def __init__(self, embedding_dim: int, max_seq_len: int = 5000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.encoding = t.tensor(calculate_positional_encoding(self.max_seq_len, self.embedding_dim))

    def forward(self, x: t.Tensor) -> t.Tensor:
        """x: shape (batch, seq_len, embedding_dim)"""
        return x + self.encoding[:x.shape[-2]]


class MLP(t.nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.model = t.nn.Sequential(
            t.nn.Linear(self.config.hidden_size, self.config.hidden_size * 4),
            t.nn.GELU(),
            t.nn.Linear(self.config.hidden_size * 4, self.config.hidden_size),
            t.nn.Dropout(self.config.dropout)
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.model(x)


class DecoderBlock(t.nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.attention = MultiheadMaskedAttention(self.config.hidden_size, self.config.num_heads)
        self.ln = t.nn.LayerNorm(self.config.hidden_size)
        self.mlp = MLP(self.config)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return x + self.ln(self.attention(x)) + self.ln(self.mlp(x))


class DecoderOnlyTransformer(t.nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.token_embedding = t.nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.positional_embedding = PositionalEncoding(self.config.hidden_size, self.config.max_seq_len)
        self.decoder_blocks = t.nn.Sequential(
            *[DecoderBlock(config) for _ in range(config.num_layers)]
        )
        self.model = t.nn.Sequential(
            t.nn.Dropout(self.config.dropout),
            self.decoder_blocks,
            t.nn.LayerNorm(self.config.hidden_size)
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        token_embedding = self.token_embedding(x)
        x = self.positional_embedding(token_embedding)
        x = self.model(x)
        return x @ self.token_embedding.weight.T


def train_transformer(data, epochs, transformer, dataset_size):
    optimiser = t.optim.Adam(transformer.parameters())

    for epoch in tqdm.tqdm(range(epochs)):
        errors = 0
        for (x, y) in data:
            x, y = x.squeeze(), y.squeeze()
            output = transformer(x)
            output = einops.rearrange(output, 'batch sequence vocab -> (batch sequence) vocab')
            y = einops.rearrange(y, 'batch sequence -> (batch sequence)')
            loss_fn = t.nn.CrossEntropyLoss()
            loss = loss_fn(output, y.long())
            argmax = t.argmax(output, dim=-1)
            for i in range(data.batch_size * config.max_seq_len):
                if argmax[i] != y[i] and i % 6 > 2:
                    errors += 1

            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

        print(f'Epoch: {epoch} Loss: {loss}, Errors: {errors}/{dataset_size * config.max_seq_len / 2}')


class CustomTextDataset(Dataset):
    def __init__(self, vocab_size: int, seq_len: int, datasize: int):
        self.datasize = datasize
        self.vocab_size = vocab_size
        self.seq_len = seq_len

    def __len__(self):
        return self.datasize

    def __getitem__(self, idx):
        input = t.randint(0, self.vocab_size, (self.seq_len,))
        target = t.flip(input, dims=(0,))
        return input, target


dataset_size = 10240
batch_size = 128
config = TransformerConfig(6, 4, 10, 96, 6)
transformer = DecoderOnlyTransformer(config).train()

text = [(t.rand(size=(1, 6)) * 10).int() for i in range(dataset_size)]
labels = [t.flip(i, dims=[-1]) for i in text]
dataset = CustomTextDataset(10, 6, dataset_size)
data = DataLoader(dataset, batch_size=batch_size)
train_transformer(data, 10, transformer, dataset_size)

transformer.eval()
for i in range(10):
    with t.inference_mode():
        input, target = text[i], labels[i]
        output = transformer(input).argmax(dim=-1)
        print(input, output, target)
