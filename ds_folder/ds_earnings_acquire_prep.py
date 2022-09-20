# notebook dependencies 
import pandas as pd
import numpy as np
import os

# visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# acquisition function
# -------------------------------------------- #

def get_earnings_df():

    '''Function to initially pull and merge the two (2) needed 
    College Scorecard tables for period 2018-2019.'''

    # checking if dataset exists
    filename = "ipums_earnings.csv"
    
    if os.path.isfile(filename):
        
        df = pd.read_csv(filename)

        print(f'dataframe shape: {df.shape}')

        return df


    # else:
        # checks local foldere for following files
        filename_01 = "ipums_earnings.csv"
        
        # created the necessary parent and child tables
        ipums_earnings = pd.read_csv(filename_01, low_memory = False)

        # cache the newly created dataframe as a .csv file
        df.to_csv("earnings_df.csv")

        # print the df shape
        print(f'dataframe shape: {df.shape}')

        # return the dataframe
        return df


# -------------------------------------------- #

# prepare function
# -------------------------------------------- #