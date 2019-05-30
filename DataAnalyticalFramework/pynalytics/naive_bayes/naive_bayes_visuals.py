"""
This module is a framework for generating visualization and numerical results using naive Bayes classification.
"""
import math
import operator
import pandas as pd
import numpy as np
import mpld3
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib import cm
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
from mpld3 import plugins
pd.options.mode.chained_assignment = None
style.use('seaborn-bright')

class Confusion_Matrix():
    """
    This represents the class for generating data visualizations and analysis using naive Bayes classification.
    """
    
    version = "1.0"

    def __init__(self):  
        """

        Initializes the use of the class and its functions 
        """
        self.y_true = None
        self.y_pred = None

    def confusion_matrix(self,y_true, y_pred,title=None,classes=None):
        """
        Generates the confusion matrix created from applying naive Bayes
        
        Parameters
        ----------
        y_true : dataframe
            the true values of the features used
        y_pred : dataframe
            the predicted values from the features used
        title : string
            the title of the confustion_matrix
        
        Returns
        -------
        figure
            confusion_matrix visualization
        """

        try:
            cm = confusion_matrix(y_true, y_pred)
            cm_norm = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])*100
            
            fig, ax = plt.subplots()
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)
            if(classes==None):
                ax.set(xticks=np.arange(cm.shape[1]),yticks=np.arange(cm.shape[0]),ylabel='True Label',xlabel='Predicted Label')
            else:
                ax.set(xticks=np.arange(cm.shape[1]),yticks=np.arange(cm.shape[0]),xticklabels=classes, yticklabels=classes,ylabel='True Label',xlabel='Predicted Label')
        
            thresh = cm.max() / 2
            for x in range(cm_norm.shape[0]):
                for y in range(cm_norm.shape[1]):
                    if(x==y):
                        ax.text(y,x,f"{cm[x,y]}({cm_norm[x,y]:.2f}%)",ha="center", va="center", fontsize=12, color="white" if cm[x, y] > thresh else "black")
                    else:
                        ax.text(y,x,f"{cm[x,y]}({cm_norm[x,y]:.2f}%)",ha="center", va="center", color="white" if cm[x, y] > thresh else "black")

            plt.title(title)
            plt.subplots_adjust(left=0)
            

            return fig

        except Exception as e:
                print(e)

    def fig_to_html(self, fig):
        
        return mpld3.fig_to_html(fig)

    def fig_show(self, fig):
        return mpld3.show(fig)


