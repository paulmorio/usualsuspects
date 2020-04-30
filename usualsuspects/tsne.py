# The T-SNE plot tool
# Author: Paul Scherer 2020

import numpy as np
from sklearn import manifold
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from utils import rand_cmap

class Quick2DTSNE(object):
	def __init__(self, X, y, perplexity, initialisation="pca", n_jobs=-1):
		"""A quick tool for creating 2D T-SNE plots with input X, 
		and optional true classifications y for coloring.
		
		From Scikit Docs:
		t-Distributed Stochastic Neighbour Embedding is a tool 
		for visualizing high-dimensional data. It converts similarities
		between data points to joint probabilities and tries to minimize
		the KL divergence between the joint probabilities of the low 
		dimensional embedding and the high-dimensional data. T-SNE has a cost
		function that is not convex. i.e. with different initializations
		we are bound to have different results.

		This tool is a shortcut to producing the T-SNE plot one finds
		so often in papers.

		Parameters
		----------
		X : array-like
			asdfasdf
		y : array-like
			asdfasdf
		perplexity : float
			The perplexity is a hyperparameter that is related to 
			the number of nearest neighbours to consider. Typically
			larger datasets need a higher perplexity to look good.
		initialization : str (default="pca")
			Possible values = ['pca', 'random'] describes the initialization.
		n_jobs : int (default=-1)
			The number of cores to use. Default is -1 which uses all 
			available system cores

		Returns
		-------
		Quick2DTSNE
		"""

		super(Quick2DTSNE, self).__init__()
		self.X = X
		self.y = y
		self.num_components = 2
		self.perplexity = perplexity
		self.initialisation = initialisation
		self.n_jobs = n_jobs

		self.tsne = manifold.TSNE(n_components=self.num_components, 
			perplexity=self.perplexity, init=self.initialisation, 
			random_state=0, n_jobs=self.n_jobs)

		self.embeddings = self.tsne.fit_transform(self.X)

	def plot_embedding(self, title=None, save_to="tsne_plot.png"):
		x_min, x_max = np.min(self.embeddings,0), np.max(self.embeddings,0)
		embs_to_plot = (self.embeddings-x_min) / (x_max-x_min)

		if y == None:
			plt.scatter(embs_to_plot[:,0], embs_to_plot[:,1])
		else:
			plt.scatter(embs_to_plot[:,0], embs_to_plot[:,1], c=y)
			
		plt.title(title)
		plt.savefig(save_to)

if __name__ == '__main__':
	from sklearn import datasets

	digits = datasets.load_digits(n_class=7)
	X = digits.data
	y = digits.target
	n_samples, n_features = X.shape

	qtsne = Quick2DTSNE(X, y, perplexity=30.0, initialisation="random")
	qtsne.plot_embedding(title="Test plot")

