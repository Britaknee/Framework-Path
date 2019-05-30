"""
This module is a framework for generating visualization and numerical results using naive Bayes classification.
"""
import math
import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib import cm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from mpl_toolkits.mplot3d import Axes3D
pd.options.mode.chained_assignment = None
style.use('seaborn-bright')

class NaiveBayes():
    """
    This represents the class for generating data visualizations and analysis using naive Bayes classification.
    """
    
    version = "1.0"

    def __init__(self):  
        """

        Initializes the use of the class and its functions 
        """
        self.X = None
        self.y = None
        self.model = None
        self.y_pred = None
        self.y_test = None
    
    def naive_bayes(self,X, y, cv_kfold=0):
        """

        Performs naive Bayes
        
        Parameters
        ----------
        X : string
            other features
        y : string
            target feature
        cv_kfold : integer
            k to be used for k-fold cross-validation
        """

        try:

            nb = GaussianNB()            
            nb.fit(X,y)
            self.y_test = y.values
            self.y_pred =nb.predict(X)         

        except Exception as e:
            print(e)


    def classification_report(self):
        """

        Returns the classification report generated from performing naive Bayes.
        
        Returns
        -------
        dictionary
            organized results of values generated from performing naive Bayes
        """

        return classification_report(self.y_test, self.y_pred, output_dict=True)