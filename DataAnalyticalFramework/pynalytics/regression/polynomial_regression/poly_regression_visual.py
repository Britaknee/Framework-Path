"""
This program is a module for generating visualizations using Polynomial Regression.
"""
import math
import operator
import mpld3
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from .poly_regression_num import PolyRegressionRes
from matplotlib import style
from matplotlib import cm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
pd.options.mode.chained_assignment = None
style.use('seaborn-bright')

class PolyRegressionVis(PolyRegressionRes):
    """
    This represents the class for generating data visualizations using Polynomial Regression.
    """
    
    version = "1.0"

    def __init__(self):  
        """

        Initializes the use of the class and its functions 
        """
        pass

    def polynomial_reg(self, dependent, independent):
        """

        Generates the visualization for polynomial regression

        Parameters
        ----------
        dependent : 2D pandas DataFrame
            the dependent(y) variable specified
        independent : 2D pandas DataFrame
            the independent(x) variable specified

        Returns
        -------
        figure
            visualization of the polynomial regression
        """

        try:
            x_column = independent.columns.values
            y_column = dependent.columns.values

            x = independent.to_numpy()
            y = dependent.to_numpy()
            
            poly= PolynomialFeatures(degree=2)
            x_poly = poly.fit_transform(x)

            model = LinearRegression()
            model.fit(x_poly, y)
            y_pred = model.predict(x_poly)

            fig = plt.figure()
            ax = fig.add_subplot(111)

            ax.scatter(x, y, s=10, color = 'red')
            sorted_axis = operator.itemgetter(0)
            sorted_zip = sorted(zip(x,y_pred), key=sorted_axis)
            x, y_pred = zip(*sorted_zip)
            ax.plot(x, y_pred, color='blue', label=self.poly_eq(dependent, independent))
            ax.legend(fontsize=9, loc="upper right")
            plt.title("Polynomial Regression of " + x_column[0] + " and " + y_column[0])
            plt.xlabel(x_column[0])
            plt.ylabel(y_column[0])
            # plt.show()
            return fig
            
        except Exception as e:
            print(e)
    
    def fig_to_html(self, fig):
        return mpld3.fig_to_html(fig)

    def fig_show(self, fig):
        return mpld3.show(fig)

