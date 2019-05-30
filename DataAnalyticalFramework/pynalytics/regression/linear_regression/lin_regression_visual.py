"""
This program is a module for generating numerical results using linear regression.
"""
import math
import operator
import mpld3
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from .lin_regression_num import LinRegressionRes
from matplotlib import style
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
pd.options.mode.chained_assignment = None
style.use('seaborn-bright')

class LinRegressionVis(LinRegressionRes):
    """
    This represents the class for generating numerical results using Linear Regression.
    """
    
    version = "1.0"

    def __init__(self):  
        """

        Initializes the use of the class and its functions 
        """
        pass

    def scatter_plot(self, dependent, independent):
        """

        Generates the visualization of the scatter plot

        Generates the 2D visualization of scatter plot given that no second independent variable is specified, else it will generate a 3D visualization

        Parameters
        ----------
        dependent : 2D pandas DataFrame
            the dependent(y) variable specified
        independent : 2D pandas DataFrame
            the independent(x) variable specified
        
        Returns
        -------
        figure
            visualization of the scatter plot
        """

        try:
            x_column = independent.columns.values
            y_column = dependent.columns.values
            if(len(x_column) == 1):
                x = independent
                y = dependent

                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.scatter(x, y, color = 'red')
                ax.set_xlabel(x_column[0])
                ax.set_ylabel(y_column[0])
                ax.axis('tight')
                plt.title("Scatter Plot of " + x_column[0] + " and " + y_column[0])
                # plt.show()
            elif(len(x_column) > 1):
                x = independent[x_column[0]]
                y = dependent[y_column[0]]
                z = independent[x_column[1]]

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(x, y, z, color = 'red')
                ax.set_xlabel(x_column[0])
                ax.set_ylabel(y_column[0])
                ax.set_zlabel(x_column[1])
                ax.axis('tight')
                plt.title("Scatter Plot of " + x_column[0] + ", " + y_column[0] + " and " + x_column[1])
                # plt.show()
            
            return fig
        except Exception as e:
            print(e)

    def linear_regression(self, dependent, independent):
        """

        Generates the visualization for simple linear regression

        Parameters
        ----------
        dependent : 2D pandas DataFrame
            the dependent(y) variable specified
        independent : 2D pandas DataFrame
            the independent(x) variable specified
        
        Returns
        -------
        figure
            visualization of the simple linear regression
        """

        try:
            x_column = independent.columns.values
            y_column = dependent.columns.values

            x = independent
            y = dependent

            lm = LinearRegression()
            model = lm.fit(x, y)
            x_new = np.linspace(x.min() - 5, x.max() + 5, 50)
            y_new = model.predict(x_new[:])

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(x_new, y_new, color = 'blue', label=self.line_eq(x, y))
            ax.legend(fontsize=9, loc="upper right")
            ax.scatter(x, y, color = 'red')
            ax.set_xlabel(x_column[0])
            ax.set_ylabel(y_column[0])
            ax.axis('tight')
            plt.title("Linear Regression of " + x_column[0] + " and " + y_column[0])
            # # plt.show()
            # mpld3.show(fig)
            return fig
        except Exception as e:
            print(e)

    def fig_to_html(self, fig):
        return mpld3.fig_to_html(fig)

    def fig_show(self, fig):
        return mpld3.show(fig)

