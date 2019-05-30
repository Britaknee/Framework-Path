"""
This program is a module for generating visualization and numerical results using k-means clustering.
"""
import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib import cm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PowerTransformer
pd.options.mode.chained_assignment = None

class Kmeans():
    """
    This represents the class for generating data visualizations and analysis using k-means clustering.
    """
    
    version = "1.0"

    def __init__(self,X ,k, rand_state=0):  
        """

        Initializes the use of the class and its functions 
        """
        self.X = X
        self.k = k
        self.model = KMeans(n_clusters=self.k,random_state=rand_state).fit(self.X)

    def sil_coef(self):
        """

        Returns the silhouette coefficient generated from using k-means clustering

        Returns
        -------
        float
            silhouette coefficient
        """

        try:
            labels = self.model.labels_
            sil_coef = silhouette_score(self.X, labels, metric='euclidean')

            return sil_coef

        except Exception as e:
            print(e)

    def centroids(self):
        """

        Returns the centroids generated from using k-means clustering
        
        Returns
        -------
        numpy array
            centroids
        """

        try:
            centroids = self.model.cluster_centers_

            return centroids

        except Exception as e:
            print(e)

    def labeled_dataset(self):
        """

        Returns the labeled dataset generated from using k-means clustering
        
        Returns
        -------
        dataframe
            original dataframe with a column containing the labels/clusters
        """

        try:
            clusters = self.model.labels_
            clusters_df = pd.DataFrame(data=clusters)
            clusters_df.columns = ['clusters']
            new_df= pd.concat([self.X,clusters_df], axis=1)

            return new_df

        except Exception as e:
            print(e)