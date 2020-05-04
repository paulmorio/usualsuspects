"""A quick example using the PCA plotter on some digits data"""

from sklearn import datasets
from usualsuspects import Quick2DPCA

digits = datasets.load_digits(n_class=7)
X = digits.data
y = digits.target

pca_plotter = Quick2DPCA(X,y)
pca_plotter.plot_embedding(title="Digits PCA Example", save_to="example_2DPCA_plot.png")
