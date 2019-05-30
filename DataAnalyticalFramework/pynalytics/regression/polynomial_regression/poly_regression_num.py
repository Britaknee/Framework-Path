"""
This program is a module for generating numerical results using Polynomial Regression.
"""
import math
import operator
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib import cm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
pd.options.mode.chained_assignment = None
style.use('seaborn-bright')

class PolyRegressionRes:
    """
    This represents the class for generating numerical results using Polynomial Regression.
    """
    
    version = "1.0"

    def __init__(self):  
        """

        Initializes the use of the class and its functions 
        """
        pass

    def get_poly_intercept(self, dependent, independent):
        """

        Returns the calculated intercept of the polynomial regression
        
        Parameters
        ----------
        dependent : 2D pandas DataFrame
            the dependent(y) variable specified
        independent : 2D pandas DataFrame
            the independent(x) variable specified

        Returns
        -------
        float
            intercept of the polynomial regression
        """

        try:
            x = independent
            y = dependent

            poly = PolynomialFeatures(degree = 2)
            x_poly = poly.fit_transform(x)   

            model = LinearRegression()
            model.fit(x_poly, y)
            intercept_arr = model.intercept_
            return round(intercept_arr[0], 4)
        except Exception as e:
            print(e)
    
    def get_poly_coeff(self, dependent, independent):
        """

        Returns a list containing the correlation coefficients of the polynomial regression
        
        Parameters
        ----------
        dependent : 2D pandas DataFrame
            the dependent(y) variable specified
        independent : 2D pandas DataFrame
            the independent(x) variable specified

        Returns
        -------
        list
            list of correlation coefficients of the polynomial regression
        """

        try:
            x = independent
            y = dependent

            poly = PolynomialFeatures(degree = 2)
            x_poly = poly.fit_transform(x)   

            model = LinearRegression()
            model.fit(x_poly, y)
            return model.coef_
        except Exception as e:
            print(e)

    def get_poly_rsquared(self, dependent, independent):
        """

        Returns the calculated coefficient of determination(R²) of the polynomial regression
        
        Parameters
        ----------
        dependent : 2D pandas DataFrame
            the dependent(y) variable specified
        independent : 2D pandas DataFrame
            the independent(x) variable specified

        Returns
        -------
        float
            calculated coefficient of determination(R²) of the polynomial regression
        """

        try:
            x = independent
            y = dependent

            poly = PolynomialFeatures(degree = 2)
            x_poly = poly.fit_transform(x)   

            model = LinearRegression()
            model.fit(x_poly, y)
            y_poly_pred = model.predict(x_poly)
            r2 = r2_score(y,y_poly_pred)
            return round(r2, 4)
        except Exception as e:
            print(e)

    def get_poly_pearsonr(self, dependent, independent):
        """

        Returns the calculated Pearson correlation coefficient of the polynomial regression
        
        Parameters
        ----------
        dependent : 2D pandas DataFrame
            the dependent(y) variable specified
        independent : 2D pandas DataFrame
            the independent(x) variable specified

        Returns
        -------
        float
            calculated Pearson correlation coefficient of the polynomial regression
        """

        try:
            x = independent
            y = dependent

            poly = PolynomialFeatures(degree = 2)
            x_poly = poly.fit_transform(x)   

            model = LinearRegression()
            model.fit(x_poly, y)
            y_poly_pred = model.predict(x_poly)
            r2 = r2_score(y,y_poly_pred)
            pearsonr = math.sqrt(r2)
            return round(pearsonr, 4)
        except Exception as e:
            print(e) 
    
    def poly_eq(self, dependent, independent):
        """

        Returns the equation of the polynomial regression

        Parameters
        ----------
        dependent : 2D pandas DataFrame
            the dependent(y) variable specified
        independent : 2D pandas DataFrame
            the independent(x) variable specified
        
        Returns
        -------
        str
            line equation of the polynomial regression
        """

        try:
            x = independent
            y = dependent

            poly = PolynomialFeatures(degree = 2)
            x_poly = poly.fit_transform(x)  

            model = LinearRegression()
            model.fit(x_poly, y)
            coef_arr = model.coef_
            intercept_arr = model.intercept_
            
            poly_equation = "y = " + str(round(coef_arr[0][2], 4)) + "x\xB2"
            
            if(coef_arr[0][1] < 0):
                poly_equation += " + (" + str(round(coef_arr[0][1], 4)) + "x" + ")"
            else:
                poly_equation += " + " + str(round(coef_arr[0][1], 4)) + "x"
            
            if(intercept_arr[0] < 0):
                poly_equation += " + (" + str(round(intercept_arr[0], 4)) + ")"
            else:
                poly_equation += " + " + str(round(intercept_arr[0], 4))
           
            return  poly_equation
        except Exception as e:
            print(e)

    def polynomial_reg_summary(self, dependent, independent):
        """

        Generates the calculated value of the coefficient of determination(R²) of the polynomial regression 

        Parameters
        ----------
        dependent : 2D pandas DataFrame
            the dependent(y) variable specified
        independent : 2D pandas DataFrame
            the independent(x) variable specified

        Returns
        ----------
        str
            calculated value of the coefficient of determination(R²) of the polynomial regression 
        """

        try:
            x_column = independent.columns.values
            y_column = dependent.columns.values

            poly_rsquared = self.get_poly_rsquared(dependent, independent)
            poly_pearsonr = self.get_poly_pearsonr(dependent, independent)
            result_str = "Pearson correlation coefficient(R) of the polynomial regression of " + x_column[0] + " and " + y_column[0] + ": " + str(poly_pearsonr)
            result_str += "\nR\xb2 of the polynomial regression of " + x_column[0] + " and " + y_column[0] + ": " + str(poly_rsquared)
            
            return result_str
        except Exception as e:
            print(e)

    def poly_reg_table(self, dependent, independent):
        """

        Generates the summary of the calculated values for Pearson Correlation Coefficient (R) and coefficient of determination(R²) of the polynomial regression, in table form

        Parameters
        ----------
        dependent : 2D pandas DataFrame
            the dependent(y) variable specified
        independent : 2D pandas DataFrame
            the independent(x) variable specified

        Returns
        -------
        pandas Dataframe
            summary of the calculated values for Pearson Correlation Coefficient (R) and coefficient of determination(R²) of the polynomial regression
        """

        try:
            x = independent
            x_column = independent.columns.values
            pearsonr = []
            rsquared = []

            for step in x_column:
                pearsonr.append(self.get_poly_pearsonr(dependent, x[[step]]))
                rsquared.append(self.get_poly_rsquared(dependent, x[[step]]))

            table_content =  {"Attribute (x)": x_column, "Pearson Correlation Coefficient (R)": pearsonr, "Coefficient of Determination (R^2)": rsquared,}
            table_df = pd.DataFrame(table_content)
            return table_df
        except Exception as e:
            print(e)