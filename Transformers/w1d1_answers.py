import math
import matplotlib.pyplot as plt


def calculate_positional_encoding(num_tokens, num_dims):
    return [[calculate_encoding(i, d, num_dims) for d in range(1, num_dims + 1)] for i in range(1, num_tokens + 1)]


def calculate_encoding(i, d, num_dims):
    theta = i / (10000 ** (2 * d / num_dims))
    return math.sin(theta) if d % 2 == 0 else math.cos(theta)


def plot_positional_encoding(num_tokens, num_dims):
    plt.imshow(calculate_positional_encoding(num_tokens, num_dims), cmap='RdBu', vmin=-1, vmax=1)
    plt.show()


plot_positional_encoding(32, 128)


