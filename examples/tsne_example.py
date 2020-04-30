"""A quick example using the T-SNE plotter on some digits data"""

from sklearn import datasets
from usualsupects import Quick2DTSNE

digits  = datasets.load_digits(n_class = 7)
X = digits.data
y = digits.target

qTsne = Quick2DTSNE(X, y, perplexity=30.0, initialisation="pca")
qTsne.plot_embedding(title="Digits TSNE Example", save_to="example_2DTSNE_plot.png")