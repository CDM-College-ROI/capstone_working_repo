# notebook dependencies 
import pandas as pd
import numpy as np
import os

# visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from scipy.stats.mstats import winsorize
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer



# ---------------------------------------------------------------- #
            ###  Preparatory Exploration Functions ###
# ---------------------------------------------------------------- #

'''This file contains:
    - functions for additional preparation of income brackets
    - train_test_split function
    - functions for capping outliers 
    - functions for imputing nulls programmatically
    
    '''



# ---------------------------------------------------------------- #
                ### Merging Income Bracket cols ###
# ---------------------------------------------------------------- #



# Create new set of income brackets merged by `institution_control`
def create_merged_income_brackets():

    income_0_30000 = [
    'other_fam_income_0_30000',
    'private_fam_income_0_30000',
    'program_fam_income_0_30000',
    'pub_fam_income_0_30000']

    income_30001_48000 = [
    'other_fam_income_30001_48000',
    'private_fam_income_30001_48000',
    'program_fam_income_30001_48000',
    'pub_fam_income_30001_48000']

    income_48001_75000 = [
    'other_fam_income_48001_75000',
    'private_fam_income_48001_75000',
    'program_fam_income_48001_75000',
    'pub_fam_income_48001_75000']

    income_75001_110000 = [
    'other_fam_income_75001_110000',
    'private_fam_income_75001_110000',
    'program_fam_income_75001_110000',
    'pub_fam_income_75001_110000']

    income_over_110000 = [
    'other_fam_income_over_110000',
    'private_fam_income_over_110000',
    'program_fam_income_over_110000',
    'pub_fam_income_over_110000']

    return income_0_30000, income_30001_48000, income_48001_75000, income_75001_110000, income_over_110000


def get_fam_income_col(df, col_lst, new_col_string):

    '''Function that creates a new family income columns from 
    existing dummy columns.'''

    df[col_lst] = df[col_lst].fillna(0)

    df[new_col_string] = df[col_lst].sum(axis = 1)

    # drop redundant columns
    df = df.drop(df[col_lst], axis = 1)

    # return the dataframe
    return df

def apply_fam_income_col(df):
    # applying the function ---
    # list of cols to collapse
    frames = [
        income_30001_48000, 
        income_48001_75000, 
        income_75001_110000, 
        income_over_110000]

    # list of new col names
    var_names = [
        'income_30001_48000', 
        'income_48001_75000', 
        'income_75001_110000', 
        'income_over_110000']

    for i in range(len(frames)):
        var_name = var_names[i]
        df = get_fam_income_col(df, frames[i], var_name)

    print(df.shape)
    df.head()

    return df

# Call 5 individual assigns within notebook
def call_bracket_function(df):
    df = get_fam_income_col(df, income_0_30000, "fam_income_0_30000")
    df = get_fam_income_col(df, income_30001_48000, "fam_income_30001_48000")
    df = get_fam_income_col(df, income_48001_75000, "fam_income_48001_75000")
    df = get_fam_income_col(df, income_75001_110000, "fam_income_75001_110000")
    df = get_fam_income_col(df, income_over_110000, "fam_income_over_110000")

    return df

# Contains all relevant functions listed above for streamlined process
def master_bracket_func(df, col_lst, new_col_string):

    income_0_30000, income_30001_48000, income_48001_75000, income_75001_110000, income_over_110000 = create_merged_income_brackets()

    df = get_fam_income_col(df, col_lst, new_col_string)

    df = apply_fam_income_col(df)

    return df



# ---------------------------------------------------------------- #
                ### Train, Validate, Test Split ###
# ---------------------------------------------------------------- #
from sklearn.model_selection import train_test_split 

def split_data(df):
    train_and_validate, test = train_test_split(
        df, 
        test_size = 0.2, 
        random_state = 123,
        stratify = df["major_category"])

    train, validate = train_test_split(
        train_and_validate,
        test_size = 0.3,
        random_state = 123,
        stratify = train_and_validate["major_category"])

    return train, validate, test

# ------------------------------------- #


# ---------------------------------------------------------------- #
                ### Handling Outliers, Nulls ###
# ---------------------------------------------------------------- #


# Handles outliers 
def percentile_capping(df, low_end, high_end):

    from scipy import stats
    from scipy.stats.mstats import winsorize

    '''Function that uses scipy's winsorize method to cap
    continuous variables at lower and higher end based on a passed 
    percentile values.'''

    l1 = df.select_dtypes(include = "number").columns.tolist()

    # dont include target variables to cap
    target_lst = [ 
                "roi_5yr",
                "roi_10yr",
                "2017",                                               
                "2018",                                                   
                "2019",
                "Grand Total",
                "avg_net_price"]

    col_lst = [col for col in l1 if col not in target_lst]

    for col in col_lst:

        stats.mstats.winsorize(
            a = df[col], 
            limits = (low_end, high_end), 
            inplace = True)
    
    return df

# _____________________________________ #


#
def train_iterative_imputer(train_df):
    # using sklearn's iterative imputer to fill-in remaining nulls
    # placeholder for continuous features
    l1 = train_df.select_dtypes(include = "number").columns.tolist()

    # dont learn from these variables
    target_lst = [ 
            "roi_5yr",
            "roi_10yr",
            "roi_20yr",
            "pct_roi_5yr",
            "pct_roi_10yr",
            "pct_roi_20yr",
            "2017",                                               
            "2018",                                                   
            "2019",
            "Grand Total",
            "avg_net_price"
    ]
    
    num_lst = [col for col in l1 if col not in target_lst]
    
    # creating the "thing"
    imputer = IterativeImputer(
            missing_values = np.nan, \
            skip_complete = True, \
            random_state = 123)
    
    # fitting the "thing" and transforming it
    imputed = imputer.fit_transform(train_df[num_lst])

        # create a new dataframe with learned imputed data
    train_df_imputed = pd.DataFrame(imputed, index = train_df.index)

    # filling in missing values from learned imputer
    train_df[num_lst] = train_df_imputed

    # return the new imputed df
    return train_df



# _____________________________________ #


# 
def impute_val_and_test(train_df, val_df, test_df):
        
        '''Function takes in all three split datasets and imputes missing values in validate and test after
        fitting on training dataset columns'''

        l1 = train_df.select_dtypes(include = "number").columns.tolist()

        target_lst = [ 
        "roi_5yr",
        "roi_10yr",
        "2017",                                               
        "2018",                                                   
        "2019",
        "Grand Total",
        "avg_net_price",
        "avg_sat_admitted",
        "ACT_score_mid"
        ]

        # recheck cols are not in target list
        num_lst = [col for col in l1 if col not in target_lst]

        # creating the sklearn imputer
        imputer = IterativeImputer(
                missing_values = np.nan, \
                skip_complete = True, \
                random_state = 123)

        # fitting the imputer
        imputed = imputer.fit(train_df[num_lst])

        # transforming values
        val_imputed = imputed.transform(val_df[num_lst])
        X_validate_imputed = pd.DataFrame(val_imputed, index = val_df.index)
        val_df[num_lst] = X_validate_imputed
        validate_imputed = val_df

        test_imputed = imputed.transform(test_df[num_lst])
        test_imputed = pd.DataFrame(test_imputed, index = test_df.index)
        test_df[num_lst] = test_imputed
        test_imputed = test_df

        # fill-in any instances of missing zip-code values
        validate_imputed["zip_code"] = validate_imputed["zip_code"].fillna(validate_imputed["zip_code"].mode()[0])
        test_imputed["zip_code"] = test_imputed["zip_code"].fillna(test_imputed["zip_code"].mode()[0])


        # returning the imputed validate and test datasets
        return validate_imputed, test_imputed