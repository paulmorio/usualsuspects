# Test module for TSNE modules classes

import os
from sklearn import datasets
from pytest import fixture


@fixture
def t_snesne():
    from usualsuspects import Quick2DTSNE

    digits = datasets.load_digits(n_class=7)
    X = digits.data
    y = digits.target
    return Quick2DTSNE(X, y, perplexity=30.0, initialisation="pca")


def test_plot_embeddings(t_snesne):
    t_snesne.plot_embedding(save_to="test/tsne_plot.png")
    assert os.path.exists("test/tsne_plot.png")
    os.remove("test/tsne_plot.png")
