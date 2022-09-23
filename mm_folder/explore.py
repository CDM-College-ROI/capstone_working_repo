# notebook dependencies 
import pandas as pd
import numpy as np
import os

# visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from scipy.stats.mstats import winsorize
from sklearn.impute import IterativeImputer
from sklearn.experimental import enable_iterative_imputer


# ---------------------------------------------------------------- #
            ###  Preparatory Exploration Functions ###
# ---------------------------------------------------------------- #


# Handles outliers 
def percentile_capping(df, low_end, high_end):

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
            "2017",                                               
            "2018",                                                   
            "2019",
            "Grand Total"
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
        "avg_net_price"
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