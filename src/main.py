""" Contains main function of project """
from src.isomap import Isomap
from src.pca import PCA
from tsne import TSNE


def main():
    """ Main function of project """
    # PCA
    print("PCA")
    pca_swiss = PCA("swiss_data.csv")
    pca_swiss.fit_data()
    pca_swiss.transform()
    pca_digits = PCA("digits.csv")
    pca_digits.fit_data()
    pca_digits.transform()

    # Isomap
    print("Isomap")
    isomap_swiss = Isomap("swiss_data.csv")
    isomap_digits = Isomap("digits.csv")
    isomap_swiss.compute_geodesics(25)      # Value of 20-30 seems to be best fit
    isomap_digits.compute_geodesics(35)     # Value of 30-40 seems to be best fit
    isomap_swiss.apply_mds()
    isomap_digits.apply_mds()

    # t-SNE
    print("t-SNE")
    tsne_digits = TSNE("digits.csv")
    tsne_digits.compute_pairwise_similarities(43)   # Value of 45 seems to be best fit
    # Values given in assignment are:
    # max_iteration:    1000
    # alpha:            0.8
    # epsilon:          500
    tsne_digits.map_data_points(1000, 0.8, 500)


if __name__ == '__main__':
    main()
