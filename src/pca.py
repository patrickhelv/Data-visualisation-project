""" File containing PCA class """
import numpy as np
import matplotlib.pyplot as plt


class PCA:
    """ Performs Primary Component Analysis """
    def __init__(self, filename):
        """ Loads csv to array """
        self.col = 0
        self.dimensions = 0     # D in the assignment
        if filename == 'swiss_data.csv':
            self.col = 2000
            self.dimensions = 3
            self.choice = 0
        elif filename == 'digits.csv':
            self.col = 5620
            self.dimensions = 64
            self.choice = 1
            self.labels = np.genfromtxt('../csv/digits_label.csv', delimiter=',').tolist()

        if self.col == 0 and self.dimensions == 0:
            print("wrong filename")
        else:
            raw_data = np.genfromtxt('../csv/' + filename, delimiter=',')   # X
            self.small_x = np.reshape(raw_data, (self.col, self.dimensions))
            self.fit = 0    # F in the assignment

    def fit_data(self):
        """ Creates fit using PCA and assigns it to self.F """
        self.small_x = self.small_x - np.mean(self.small_x, axis=0)
        covmatrix = np.cov(self.small_x.T)

        [eigenvalues, eigenvectors] = np.linalg.eig(covmatrix)

        eig_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:, i]) for i in range(len(eigenvalues))]
        eig_pairs.sort(key=lambda x: x[0], reverse=True)
        self.fit = np.hstack((eig_pairs[0][1].reshape(self.dimensions, 1),
                              eig_pairs[1][1].reshape(self.dimensions, 1)))

    def transform(self):
        """ Transforms data using fit (self.fit). Plots result. """
        small_y = self.fit.T @ (self.small_x.T)
        if self.choice == 0 :
            plt.scatter(small_y[1, :], small_y[0, :],
                        c=np.arange(self.col), cmap='gist_rainbow', s=7)
        elif self.choice == 1:
            plt.scatter(small_y[0, :], small_y[1, :],
                        c=self.labels, cmap='jet', s=7, marker=".")
        plt.show()
