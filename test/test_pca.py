# Test module for PCA module classes

import os
from sklearn import datasets
from pytest import fixture

@fixture
def t_pcapca():
	from usualsuspects import Quick2DPCA

	digits = datasets.load_digits(n_class=7)
	X = digits.data
	y = digits.target
	return Quick2DPCA(X,y)

def test_plot_embeddings(t_pcapca):
	t_pcapca.plot_embedding(save_to="test/pca_plot.png")
	assert os.path.exists("test/pca_plot.png")
	os.remove("test/pca_plot.png")