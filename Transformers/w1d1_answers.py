import math
import matplotlib.pyplot as plt
import numpy as np


def calculate_positional_encoding(num_tokens, num_dims):
    return [[calculate_encoding(i, d, num_dims) for d in range(1, num_dims + 1)] for i in range(1, num_tokens + 1)]


def calculate_encoding(i, d, num_dims):
    theta = i / (10000 ** (2 * d / num_dims))
    return math.sin(theta) if d % 2 == 0 else math.cos(theta)


def plot_positional_encoding(num_tokens, num_dims):
    plt.imshow(calculate_positional_encoding(num_tokens, num_dims), cmap='RdBu', vmin=-1, vmax=1)
    plt.show()


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def get_dot_product_graph(num_tokens, num_dims):
    pe = calculate_positional_encoding(num_tokens, num_dims)
    dots = np.array([cosine_similarity(pe[i], pe[j]) for i in range(len(pe)) for j in range(len(pe))])
    min_dots, max_dots = min(dots), max(dots)
    plt.imshow(dots.reshape(num_tokens, num_tokens), cmap="Blues", vmin=min_dots, vmax=max_dots)
    plt.show()


get_dot_product_graph(32, 128)
