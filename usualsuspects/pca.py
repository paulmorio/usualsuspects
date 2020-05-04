# The PCA plot tool
# Author: Paul Scherer 2020

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class Quick2DPCA(object):
    """A quick tool for crating 2D PCA plots with input X,
    and optional true classification y for colouring

    From SciKit Docs:
    Linear dimensionality reduction using SVD of the data
    to project it to a lower dimensional space. The input data
    is centered but not scaled for each feature before applying the
    SVD

    This tool is a shortcut to producing PCA plots

    """

    def __init__(self, X, y):
        super(Quick2DPCA, self).__init__()
        self.X = X
        self.y = y
        self.num_components = 2

        self.embeddings = PCA(n_components=2).fit_transform(X)

    def plot_embedding(self, title=None, save_to="pca_plot.png"):
        """Creates and saves a matplotlib pyplot of the first 2
        eigenvectors of the PCA operation on X

        Parameters
        ----------
        title : str
                Optional title for the plot
        save_to : str
                path and file_name where the image will be saved.
                Extension is contextual but expected behaviour is to use .png

        Returns
        -------
        None
                Saves a PNG image of the plot to the path in `save_to`

        """
        x_min, x_max = np.min(self.embeddings, 0), np.max(self.embeddings, 0)
        embs_to_plot = (self.embeddings-x_min) / (x_max - x_min)

        # Check if class labels have been given for colouring
        if len(self.y)==self.X.shape[0] or len(self.y)==self.X.shape[1]:
            plt.scatter(embs_to_plot[:,0], embs_to_plot[:,1], c=self.y)
            plt.xlabel('First Eigenvector')
            plt.ylabel('Second Eigenvector')
        else:
            plt.scatter(embs_to_plot[:,0], embs_to_plot[:,1])
            plt.xlabel('First Eigenvector')
            plt.ylabel('Second Eigenvector')

        plt.title(title)
        plt.savefig(save_to)

if __name__ == '__main__':
    from sklearn import datasets

    digits = datasets.load_digits(n_class=7)
    X = digits.data
    y = digits.target

    n_samples, n_features = X.shape

    pcaplot = Quick2DPCA(X,y)
    pcaplot.plot_embedding(title="Test plot")