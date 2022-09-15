""" File containing TSNE class """
import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt

from src import util


class TSNE:
    """ Class for performing Student t-Distributed Stochastic Neighbor Embedding """
    def __init__(self, filename):
        self.filename = filename
        self.raw = util.load_csv_to_array(filename)
        self.nr_data_points = self.raw.shape[0]
        self.hd_similarity_matrix = None    # p as described in assignment

    def compute_pairwise_similarities(self, k):
        """
        Computes pairwise similarities between the raw data points.
        Creates a new matrix filled with zeroes, except for at the
        indexes of the k+1 nearest points in the distance matrix, where
        the value is set to 1. Diagonal is then set to zero
        (in other words distance from point to itself is not included).

        Sets the hd_similarity_matrix of self to be the resulting matrix.
        """
        self.hd_similarity_matrix = util.calculate_euclidean_distances(self.raw)
        self.hd_similarity_matrix = util.reduce_matrix(self.hd_similarity_matrix, k)

        self.hd_similarity_matrix = \
            (self.hd_similarity_matrix + np.swapaxes(self.hd_similarity_matrix, 0, 1) > 0)\
                .astype(float)

    def map_data_points(self, max_iteration, alpha, epsilon):
        """ Maps data points. """

        # divide each point by sum of values
        stand_hd_similarity_matrix = self.hd_similarity_matrix / \
                                     np.sum(self.hd_similarity_matrix)  # P
        dynamic_stand_hd_similarity_matrix = 4 * stand_hd_similarity_matrix

        # Sample 2D data points from normal distribution
        sampled_two_d_points = normal(0, 10e-4, (2, self.nr_data_points))
        util.save_array_to_csv(sampled_two_d_points, "sampled_2d_points.csv")

        # Or load previously sampled 2D points for consistency
        sampled_two_d_points = util.load_csv_to_array("sampled_2d_points.csv")

        # Initialize variables
        gain = np.ones((2, self.nr_data_points))        # g in assignment
        change = np.zeros((2, self.nr_data_points))     # delta in assignment
        dynamic_alpha = 0.5

        for i in range(1, max_iteration):
            print("Iteration " + str(i))

            if i == 250:
                dynamic_alpha = alpha   # Optimisation trick

            # Find similarity matrix of 2D points
            two_d_similarity_matrix = util.calculate_euclidean_distances(
                np.swapaxes(sampled_two_d_points, 0, 1))
            two_d_similarity_matrix = 1 / (
                    1 + np.square(two_d_similarity_matrix))  # q as described in assignment

            # divide each point by sum of values
            stand_two_d_similarity_matrix = two_d_similarity_matrix / \
                                            np.sum(two_d_similarity_matrix)  # Q

            if i == 100:
                dynamic_stand_hd_similarity_matrix = \
                    stand_hd_similarity_matrix     # Optimisation trick

            capital_y = np.swapaxes(sampled_two_d_points, 0, 1)
            capital_g = (dynamic_stand_hd_similarity_matrix -
                 stand_two_d_similarity_matrix) * two_d_similarity_matrix
            capital_s = np.diag(np.sum(capital_g, axis=1))
            gradient = 4 * (capital_s - capital_g) @ capital_y

            # print(gradient.shape)
            gradient = np.swapaxes(gradient, 0, 1)

            # Update gain
            gain[np.sign(gradient) != np.sign(change)] += 0.2
            gain[np.sign(gradient) == np.sign(change)] *= 0.8
            gain[gain < 0.01] = 0.01

            # Update change
            change = dynamic_alpha * change - epsilon * gain * gradient

            # Update 2D points
            sampled_two_d_points += change

            print(sampled_two_d_points)

        # Plotting mapped points
        if self.filename == "digits.csv":
            labels = util.load_csv_to_array("digits_label.csv").tolist()
            points_to_plot = np.swapaxes(sampled_two_d_points, 0, 1)
            plt.scatter(points_to_plot[:, 0], points_to_plot[:, 1],
                        c=labels, cmap='tab10', s=10, marker=".")
            cbar = plt.colorbar()
            cbar.set_label("Number labels")
        else:
            plt.scatter(sampled_two_d_points[:, 0], sampled_two_d_points[:, 1],
                        s=10, marker=".")
        plt.show()
