"""
This program is a module for generating numerical results using linear regression.
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
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
pd.options.mode.chained_assignment = None
style.use('seaborn-bright')

class LinRegressionRes:
    """
    This represents the class for generating numerical results using Linear Regression.
    """
    
    version = "1.0"

    def __init__(self):  
        """

        Initializes the use of the class and its functions 
        """
        pass

    def get_slope(self, dependent, independent):
        """

        Returns the slope of the regression

        Returns the calculated slope(m) of the simple linear regression given that there is no second independent variable specified, else it will return a list containing the calculated slope(m) of the multiple linear regression
        
        Parameters
        ----------
        dependent : pandas DataFrame
            the dependent(y) variable specified
        independent : pandas Dataframe
            independent(x) variable specified used for linear regression
        
        Returns
        -------
        float
            calculated slope(m) of the simple linear regression
        """

        try:
            x = independent
            y = dependent

            x = sm.add_constant(x)
            model = sm.OLS(y, x).fit() 
            coef_df = model.params
            return round(coef_df[1], 4)
        except Exception as e:
            print(e)

    def get_intercept(self, dependent, independent):
        """

        Returns the calculated intercept of the simple linear regression
        
        Parameters
        ----------
        dependent : 2D pandas DataFrame
            the dependent(y) variable specified
        independent : 2D pandas DataFrame, 2D 
            the independent(x) variable specified

        Returns
        -------
        float
            intercept of the simple linear regression
        """

        try:
            x = independent
            y = dependent

            lm = LinearRegression()
            lm.fit(x, y)
            b = lm.intercept_
            return round(b[0], 4)
        except Exception as e:
            print(e)

    def get_rsquare(self, dependent, independent):
        """

        Returns the calculated coefficient of determination(R²) of the regression

        Returns the calculated coefficient of determination(R²) of the simple linear regression given that there is no second independent variable specified, else it will return the calculated coefficient of determination(R²) of the mulltiple linear regression 
        
        Parameters
        ----------
        dependent : pandas DataFrame
            the dependent(y) variable specified
        independent : pandas DataFrame, 2D if multiple
            independent(x) variable specified used for linear regression
        
        Returns
        -------
        float
            coefficient of determination(R²) of the regression
        """

        try:
            x = independent
            y = dependent

            x = sm.add_constant(x)
            model = sm.OLS(y, x).fit()
            return round(model.rsquared, 4)
        except Exception as e:
            print(e)

    def get_adj_rsquare(self, dependent, independent):
        """
        
        Returns the calculated adjusted coefficient of determination(R²) of the regression

        returns the calculated adjusted coefficient of determination(R²) of the simple linear regression given that there is no second independent variable specified, else it will return the calculated adjusted coefficient of determination(R²) of the mulltiple linear regression 
        
        Parameters
        ----------
        dependent: pandas DataFrame
            the dependent(y) variable specified
        independent : pandas DataFrame, 2D if multiple
            the independent(x) variable specified

        Returns
        -------
        float
            calculated adjusted coefficient of determination(R²) of the regression
        """
        try:
            x = independent
            y = dependent

            x = sm.add_constant(x)           
            model = sm.OLS(y, x).fit()
            return round(model.rsquared_adj, 4)
        except Exception as e:
            print(e)
        
    def get_pearsonr(self, dependent, independent):
        """

        Returns the calculated Pearson correlation coefficient of the regression

        Returns the calculated Pearson correlation coefficient of the simple linear regression given that there is no second independent variable specified, else it will return the calculated coefficient of determination(R²) of the mulltiple linear regression 
        
        Parameters
        ----------
        dependent : pandas DataFrame
            the dependent(y) variable specified
        independent : pandas DataFrame, 2D if multiple
            the independent(x) variable specified
        
        Returns
        -------
        float
            Pearson correlation coefficient of the regression
        """

        try:
            x = independent
            y = dependent

            x = sm.add_constant(x)
            model = sm.OLS(y, x).fit()
            r2 = model.rsquared
            pearsonr = math.sqrt(r2)
            return round(pearsonr, 4)
        except Exception as e:
            print(e)

    def get_pvalue(self, dependent, independent):
        """

        Returns the calculated P-value/s of the regression

        Returns the dataframe containing calculated P-value/s of the simple linear regression given that there is no second independent variable specified, else it will return the calculated P-value/s of the mulltiple linear regression 
        
        Parameters
        ----------
        dependent : pandas DataFrame
            the dependent(y) variable specified
        independent : pandas DataFrame, 2D if multiple
            the independent(x) variable specified
        
        Returns
        -------
        pandas Dataframe
            dataframe containing the P-value/s of the regression
        """

        try:
            x_column = independent.columns.values
            x = independent
            y = dependent

            x = sm.add_constant(x)           
            model = sm.OLS(y, x).fit()
            pvalue = model.pvalues
            return round(pvalue[x_column[0]], 4)
        except Exception as e:
            print(e)

    def line_eq(self, dependent, independent):
        """

        Returns the line equation of the simple linear regression 
        
        Parameters
        ----------
        dependent : 2D pandas DataFrame
            the dependent(y) variable specified
        independent : 2D pandas DataFrame
            the independent(x) variable specified
        
        Returns
        -------
        str
            line equation of the simple linear regression
        """

        try:
            m = self.get_slope(dependent, independent)
            b = self.get_intercept(dependent, independent)
            lin_equation = "y = " + str(m) + "x "
            if(b < 0):
                lin_equation += "+ (" + str(m) + ")"
            else:
                lin_equation += "+ " + str(b)
            
            return lin_equation
        except Exception as e:
            print(e)

    def linear_reg_summary(self, dependent, independent):
        """

        Generates the calculated statistical values of the regression

        Generates the calculated statistical values for the linear regression such as the standard error, coefficient of determination(R²) and p-value, in table form

        Parameters
        ----------
        dependent : 2D pandas DataFrame
            the dependent(y) variable specified
        independent : 2D pandas DataFrame
            the independent(x) variable specified
        
        Returns
        -------
        statsmodels.summary
            table summary containing the calculated statistical values of the regression
        """

        try:
            x = independent
            y = dependent

            x = sm.add_constant(x)
            model = sm.OLS(y, x).fit()
            return model.summary()
        except Exception as e:
            print(e)
    
    def lin_regression_table(self, dependent, independent):

        """

        Generates the summary of the calculated values for P-value, correlation coefficient, coefficient of determination(R²) and adjusted coefficient of determination of regression, in table form

        Parameters
        ----------
        dependent : 2D pandas DataFrame
            the dependent(y) variable specified
        independent : 2D pandas DataFrame
            the independent(x) variable specified

        Returns
        -------
        pandas Dataframe
            summary of the calculated values for P-value, correlation coefficient, coefficient of determination(R²) and adjusted coefficient of determination of regression
        """

        try:
            x = independent
            x_column = independent.columns.values
            coeff_det = []
            adj_coeff_det = []
            pearsonr = []
            pvalue = []
            
            for step in x_column:
                pvalue_df = self.get_pvalue(dependent, x.loc[: , [step]])
                pvalue.append(pvalue_df)
                coeff_det.append(self.get_rsquare(dependent, x[step]))
                adj_coeff_det.append(self.get_adj_rsquare(dependent, x[step]))
                pearsonr.append(self.get_pearsonr(dependent, x[step]))

            table_content =  {"Attribute (x)": x_column, "P-Value": pvalue, "Coefficient of Determination (R^2)": coeff_det, "Adjusted Coefficient of Determination (R^2)": adj_coeff_det, "Pearson Correlation Coefficient (R)": pearsonr}
            table_df = pd.DataFrame(table_content)
            return table_df
        except Exception as e:
            print(e)

