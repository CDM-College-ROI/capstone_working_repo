# notebook dependencies 
import pandas as pd
import numpy as np
import os

# mathematical modules
import math
from math import sqrt
import scipy.stats as stats

# visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats.mstats import winsorize

# !iterative imputer must follow this import sequence!
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor, LogisticRegression

# ------------------------------------------------------------------------------- #
        # Modeling functions for Clustering on College Scorecard Dataset
# ------------------------------------------------------------------------------- #


# ------------------------------------------------ #
        # Binning and Dummy Var Functions #
# ------------------------------------------------ #

def get_share_bins(train_df, val_df, test_df):

    # train transformations
    train_df['share_entering_ft_binned'] = pd.qcut(
    train_df['share_entering_students_first_ft'], \
    q = 4, \
    labels = ["below_average", "average", "above_average", "highest_average"])

    train_df['admission_rate_binned'] = pd.qcut(
    train_df['admission_rate'], \
    q = 5, \
    labels = ["very_competitive", "somewhat_competitive", "competitive", "average_acceptance", "above_average_acceptance"])

    train_df["SAT_binned"] = pd.qcut(
    train_df['avg_sat_admitted'], \
    q = 4, \
    labels = ["average_sat", "above_average_sat", "competitive_sat", "very_competitive_sat"])

    # validate transformations
    val_df['share_entering_ft_binned'] = pd.qcut(
        val_df['share_entering_students_first_ft'], \
        q = 4, \
        labels = ["below_average", "average", "above_average", "highest_average"])

    val_df['admission_rate_binned'] = pd.qcut(
    val_df['admission_rate'], \
    q = 5, \
    labels = ["very_competitive", "somewhat_competitive", "competitive", "average_acceptance", "above_average_acceptance"])

    val_df["SAT_binned"] = pd.qcut(
    val_df['avg_sat_admitted'], \
    q = 4, \
    labels = ["average_sat", "above_average_sat", "competitive_sat", "very_competitive_sat"])

    # test transformations
    test_df['share_entering_ft_binned'] = pd.qcut(
        test_df['share_entering_students_first_ft'], \
        q = 4, \
        labels = ["below_average", "average", "above_average", "highest_average"])

    test_df['admission_rate_binned'] = pd.qcut(
    test_df['admission_rate'], \
    q = 5, \
    labels = ["very_competitive", "somewhat_competitive", "competitive", "average_acceptance", "above_average_acceptance"])

    test_df["SAT_binned"] = pd.qcut(
    test_df['avg_sat_admitted'], \
    q = 4, \
    labels = ["average_sat", "above_average_sat", "competitive_sat", "very_competitive_sat"])


    print(f'train shape: {train_df.shape}')
    print(f'validate shape: {val_df.shape}')
    print(f'test shape: {test_df.shape}')

    # return the transformed datasets
    return train_df, val_df, test_df


# ------------------------------------------------ #


def get_dummy_dataframes(train_df, val_df, test_df):
    '''Function creates new dataframes with dummy variables for modeling'''

    # train dataset
    train_dummy = pd.get_dummies(
        data = train_df, 
        columns = [
        'major_category',
        'share_entering_ft_binned',
        'institution_control',
        'us_region',
        'admission_rate_binned',
        'SAT_binned'],
        drop_first = False, 
        dtype = bool)

    # validate dataset
    val_dummy = pd.get_dummies(
        data = val_df, 
        columns = [
        'major_category',
        'share_entering_ft_binned',
        'institution_control',
        'us_region',
        'admission_rate_binned',
        'SAT_binned'],
        drop_first = False, 
        dtype = bool)

    # test dataset
    test_dummy = pd.get_dummies(
        data = test_df, 
        columns = [
        'major_category',
        'share_entering_ft_binned',
        'institution_control',
        'us_region',
        'admission_rate_binned',
        'SAT_binned'],
        drop_first = False, 
        dtype = bool)

    print(f'train shape: {train_dummy.shape}')
    print(f'validate shape: {val_dummy.shape}')
    print(f'test shape: {test_dummy.shape}')

    # returning the datasets
    return train_dummy, val_dummy, test_dummy


def get_cluster_dummy(train_df, val_df, test_df):
    '''After clustering, this function intends to create dummy variables for
    clusters to assist in modeling'''

    # train dataset
    train_dummy = pd.get_dummies(data = train_df, columns = [
        'admission_clusters_5yr',
        'control_clusters_5yr',
        'region_clusters_5yr',
        'ft_clusters_5yr',
        'major_clusters_5yr',
        'sat_clusters_5yr'
        ],
        drop_first = False, 
        dtype = bool)

    # validate dataset
    validate_dummy = pd.get_dummies(data = val_df, columns = [
        'admission_clusters_5yr',
        'control_clusters_5yr',
        'region_clusters_5yr',
        'ft_clusters_5yr',
        'major_clusters_5yr',
        'sat_clusters_5yr'],
        drop_first = False, 
        dtype = bool)

    # test dataset
    test_dummy = pd.get_dummies(data = test_df, columns = [
        'admission_clusters_5yr',
        'control_clusters_5yr',
        'region_clusters_5yr',
        'ft_clusters_5yr',
        'major_clusters_5yr',
        'sat_clusters_5yr'],
        drop_first = False, 
        dtype = bool)

    # returning the new dataframes
    return train_dummy, validate_dummy, test_dummy


# ------------------------------------------------ #
             # Modeling Functions #
# ------------------------------------------------ #

def establish_baseline(train, validate):
    '''function that establishes a baseline for train and validate - 
    will be used for model comparison'''

    baseline_train = round(train["roi_5yr"].mean(), 4)
    baseline_val = round(validate["roi_5yr"].mean(), 4)

    train['baseline'] = baseline_train
    validate['baseline'] = baseline_val

    train_rmse = sqrt(mean_squared_error(train.roi_5yr, train.baseline))
    validate_rmse = sqrt(mean_squared_error(validate.roi_5yr, validate.baseline))

    print('Train baseline RMSE: {:.2f}'.format(train_rmse))
    print('Validate baseline RMSE: {:.2f}'.format(validate_rmse))

    train = train.drop(columns = "baseline")
    validate = validate.drop(columns = "baseline")

    print()
    print(f'train shape: {train.shape}')
    print(f'validate shape: {validate.shape}')

    return train, validate

def recursive_feature_eliminate(X_train, y_train, number_of_top_features):
    '''Creating a recursive feature eliminate function'''
    
    # initialize the ML algorithm
    lm = LinearRegression()

    rfe = RFE(lm, n_features_to_select = number_of_top_features)

    # fit the data using RFE
    rfe.fit(X_train, y_train) 

    # get the mask of the columns selected
    feature_mask = rfe.support_

    # get list of the column names
    rfe_features = X_train.iloc[:,feature_mask].columns.tolist()

    # view list of columns and their ranking
    # get the ranks using "rfe.ranking" method
    variable_ranks = rfe.ranking_

    # get the variable names
    variable_names = X_train.columns.tolist()

    # combine ranks and names into a df for clean viewing
    rfe_ranks_df = pd.DataFrame({'Feature': variable_names, 'Ranking': variable_ranks})

    # sort the df by rank
    return rfe_ranks_df.sort_values('Ranking')


def get_melted_table(df):
  '''Function to create a melted model column to help with plotting'''

  baseline = df["baseline_mean_predictions"].median()
    
  df1 = df[[
    'roi_5yr',
    'linear_predictions', 
    'lars_predictions', 
    'tweedie_predictions']]
  
  melt_df = df1.melt("roi_5yr", var_name = 'cols',
                  value_name = 'vals')
  
  melt_df["baseline_prediction"] = baseline
  melt_df["residual"] = melt_df["roi_5yr"] - melt_df['vals']

  return melt_df


def plot_model_residuals(melt_df):
    '''Model Residual (error) Plot'''

    plt.figure(figsize=(16,8))
    plt.axhline(label='_nolegend_', 
                color = 'purple',
                ls = ':')

    ax = sns.scatterplot(data = melt_df.sample(100, random_state = 123), 
                x = 'roi_5yr', 
                y = 'residual',
                hue = 'cols',
                y_jitter = .5,
                x_jitter = .5,
                s = 50)

    legend = ax.legend()
    plt.legend()
    plt.xlabel('Actual Actual 5YR Return on Investment')
    plt.ylabel('Residual - 5YR Return on Investment')
    plt.title('Model Residual Plot')
    plt.show()

def plot_models(melt_df):
        '''Plotting actual 5yr roi, baseline_predictions, and model predictions'''
        plt.figure(figsize = (16, 8))
        plt.plot(melt_df['roi_5yr'], melt_df['baseline_prediction'], alpha=0.5,
        color='gray', ls = ':', label='_nolegend_')

        plt.plot(melt_df['roi_5yr'], melt_df['roi_5yr'], alpha=0.5,
        color='blue', label = '_nolegend_')

        ax = sns.scatterplot(data = melt_df.sample(300, random_state = 123), 
                x = 'roi_5yr', 
                y = "vals", 
                hue = 'cols',
                y_jitter = .5,
                x_jitter = .5,
                s = 50)

        legend = ax.legend()
        plt.legend()
        plt.xlabel("Actual 5YR Return on Investment")
        plt.ylabel('Predicted 5YR Return on Investment')
        plt.title('Actual 5YR Return on Investment vs Predicted Cluster 5YR Return on Investment')
        plt.show()