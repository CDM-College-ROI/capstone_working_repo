# notebook dependencies 

import pandas as pd
import numpy as np
import os

# visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# time/sleep modules
from time import sleep

# random module for sleep combination
import random

import vaex as vx

from time import time



def get_majors_df():

    '''Function to initially pull and merge the two (2) needed 
    College Scorecard tables for period 2018-2019.'''

    # checking if dataset exists
    filename = "majors_table.csv"
    
    if os.path.isfile(filename):
        
        df = pd.read_csv(filename)

        print(f'dataframe shape: {df.shape}')

        return df

    else:
        # checks local foldere for following files
        filename_01 = "FieldOfStudyData1718_1819_PP.csv"
        filename_02 = "MERGED2018_19_PP.csv"
        
        # created the necessary parent and child tables
        df_parent = pd.read_csv(filename_01, low_memory=False)
        df_child = pd.read_csv(filename_02, low_memory=False)

        df_parent["UNITID"] = df_parent["UNITID"].astype("Int32", errors='ignore')
        df_child["UNITID"] = df_child["UNITID"].astype("Int32", errors='ignore')

        df = df_parent.merge( 
        df_child,
        how = "left",
        on = "UNITID",
        copy = False
        )
        # cache the newly created dataframe as a .csv file
        df.to_csv("majors_table.csv")
        # print the df shape
        print(f'dataframe shape: {df.shape}')

        # return the dataframe
        return df