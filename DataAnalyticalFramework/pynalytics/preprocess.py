"""
This program is a module for processing the data specified.
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression

class Preprocessing():
    """
    This represents the class for processing the contents of a specified data in preparation for applying data analytics techniques.
    """
    version = "1.0"
    def __init__(self):  
        """

        Initializes the use of the class and its functions 
        
        If a dataframe input is specified, it initializes that dataframe as the source of data that will be used throughout the program

        Parameters
        ----------
        df_input : str, optional
            the data frame where the visualizations will be based from
        """
        
    def locate(self, column, cond_inp):
        """

        Returns a boolean Series based on locating certain rows of a specified column which satisfies a specified condition
        
        Parameters
        ----------
        column : str
            the column of the dataframe where the rows will located at and compared with cond_inp
        cond_inp : str
            the conditional input that will be compared with the contents of the rows of the specified column

        Returns
        -------
        boolean Series
            series containing rows of a specified column which satisfies a specified condition
        """

        try:
            return self.df.loc[self.df[column] == cond_inp]
        except Exception as e:
            print(e)
  
    def group_frame_from(self, identifier, from_int, to_int, df_input=None):
        """

        Returns a Series which is grouped based on a certain identifier and a specific column range

        Parameters
        ----------
        identifier : str
            the identifier which will serve as the basis for grouping the dataframe
        from_int : int
            the index start of the column/s to be grouped
        to_int: int
            the index end of the column/s to be grouped
        df_input: pandas Dataframe, optional
            custom dataframe to be grouped

        Returns
        -------
        Series
            series containing the grouped rows based on a certain identifier and a specific column range
        """

        try:
            if df_input is None:
                first_df = self.df.groupby([identifier],as_index=False)[self.df.columns[from_int:to_int]].sum()
                return first_df
            else:
                first_df = df_input.groupby([identifier],as_index=False)[df_input.columns[from_int:to_int]].sum()
                return first_df
        except Exception as e:
            print(e)
      
    def group_frame_except(self, identifier, from_int, to_int, df_input=None):
        """

        Returns a Series which is grouped based on a certain identifier and a specific column range wherein the outliers of the specified index end are excluded from grouping but will still be part of the Series returned

        Parameters
        ----------
        identifier : str
            the identifier which will serve as the basis for grouping the dataframe
        from_int : int
            the index start of the column/s to be grouped
        to_int : int
            the index end of the column/s to be grouped
        df_input : pandas Dataframe, optional
            custom dataframe to be grouped
        
        Returns
        -------
        Series
            series containing the grouped rows based on a certain identifier and a specific column range with the exception of the outliers
        """

        try:
            if df_input is None:
                first_df = self.df.groupby([identifier],as_index=False)[self.df.columns[from_int:to_int]].sum()
                second_df = self.df.iloc[: , to_int:]
                return first_df.join(second_df) 
            else:
                first_df = df_input.groupby([identifier])[df_input.columns[from_int:to_int]].sum()
                second_df = df_input.iloc[: , to_int:]
                return first_df.join(second_df) 
        except Exception as e:
            print(e)

    def extract_row(self, column, identifier, df_input=None):
        """

        Returns a boolean Series containing the content of rows based on the specified column which matches a specific identifier
        
        Parameters
        ----------
        column : str
            the column of the dataframe where the rows will extracted at
        identifier : str
            the identifier of the rows to be extracted
        df_input : pandas Dataframe, optional
            custom dataframe to be grouped

        Returns
        -------
        boolean Series
            series containing rows of a specific column which matches a specific identifier
        """

        try:
            if df_input is None:
                return self.df.loc[self.df[column] == identifier]
            else:
                return df_input.loc[df_input[column] == identifier]  
        except Exception as e:
            print(e)

    def scaler(self, columns=None, minmax=(0,1)):
        scaler = MinMaxScaler(feature_range= minmax)
        if(self.target==None):
            self.df = scaler.fit_transform(self.df.astype(float))
        else:
            for col in self.df.columns:
                if(col !=self.target):
                    self.df[col] = scaler.fit_transform(self.df[[col]].astype(float))

    def bin(self,df,n, strat="uniform"):

        valid_strategy = ('uniform', 'quantile', 'kmeans')
        if strat not in valid_strategy:
            raise ValueError("Valid options for 'bin_strat' are {}. "
                             "Got strategy={!r} instead."
                             .format(valid_strategy, strat))
        est = KBinsDiscretizer(n_bins=n, encode='ordinal', strategy=strat)

        return est.fit_transform(df)