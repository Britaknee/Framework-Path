"""
This program is a module for generating visualization and numerical results using k-means clustering.
"""
import math
import operator
import mpld3
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
style.use('seaborn-bright')

class Centroid_Chart():
    """
    This represents the class for generating data visualizations and analysis using K-means clustering.
    """
    
    version = "1.0"

    def __init__(self):  
        """

        Initializes the use of the class and its functions 
        """
        self.centroids = None
        self.x_labels = None

    def centroid_chart(self, centroids, x_labels=None, title=None):
        """

        Generates the centroid chart created from performing k-means clustering
        
        Parameters
        ----------
        centroids : numpy array
            the value of the centroids
        x_labels : array
            the columns 
        title : string
            the title of the centroid chart
        
        Returns
        -------
        figure
            centroid chart visualization
        """

        ##centroids: numpy array
        try:                              
            fig = plt.figure()
            ax = fig.subplots()
            for k in range(centroids.shape[0]):
                plt.plot(range(centroids.shape[1]),centroids[k], label=str(k)+": "+str(centroids[k]))

            if (x_labels.all()!=None):
                print(x_labels)
                plt.xticks(range(centroids.shape[1]),x_labels)
            
            plt.xlabel("Features")
            plt.ylabel("Location")
            plt.setp( ax.xaxis.get_majorticklabels(), rotation=-15, ha="left" )
            plt.legend(loc="upper right",title='Clusters (Centroids)')
            plt.title(title)
            plt.tight_layout()
            return fig

        except Exception as e:
                print(e)

    def fig_to_html(self, fig):
        return mpld3.fig_to_html(fig)

    def fig_show(self, fig):
        return mpld3.show(fig)


class Scatter_Matrix():

        def __init__(self):  

            """
            Initializes the use of the class and its functions 
            """
            self.df = None
            self.clusters_column = None

        def scatter_matrix(self, df, clusters_column=None, cmap='Set1', title=None):
            """

            Generates the centroid chart created from performing k-means clustering
        
            Parameters
            ----------
            df : dataframe
                the dataframe to be used
            clusters_column : string
                the name of the column 
            cmap : 'Set1'
                the color map set
            title : string
                the title of the centroid chart
        
            Returns
            -------
            figure
                matrix containing the clustered scatter plots
            """

            ##centroids: numpy array
            try:
                features = df.shape[1]  if clusters_column==None else df.shape[1]-1
                fig = plt.figure()
                fig.subplots_adjust(hspace=0.5)
                axctr = 1
                for y in range(0,features):
                    for x in range(0,features):
                        ax = fig.add_subplot(features, features, axctr)
                        axctr = axctr+1
                        for c in range(len(df[clusters_column].unique())):
                            temp_df = df[df[clusters_column] == c]
                            ax.scatter(temp_df[df.columns[y]], temp_df[df.columns[x]], label=c, cmap=cmap)
                        if(x==0):
                            ax.set_ylabel(df.columns[y])
                        if(y==features-1):
                            ax.set_xlabel(df.columns[x])
                        ax.axis('tight')
                        if(y==0 and x==features-1):
                                ax.legend(title='Clusters', loc='upper center',bbox_to_anchor=(0.5, (1+(features/7))), ncol=len(df[clusters_column].unique()))

                plt.suptitle(title)
                # fig.legend(labels=('label1', 'label2', 'label3'),loc='upper right')

                return fig

            except Exception as e:
                    print(e)

        def fig_to_html(self, fig):
            return mpld3.fig_to_html(fig)

        def fig_show(self, fig):
            return mpld3.show(fig)