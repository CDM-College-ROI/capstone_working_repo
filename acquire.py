


#### Acquire.py File ####

''' Houses function to acquire necessary datasets and merges them into one complete df'''

def get_mass_majors_df():
    '''Function to initially pull and merge the two (2) needed
    College Scorecard tables for period 2018-2019.'''
    filename_01 = "FieldOfStudyData1718_1819_PP.csv"
    filename_02 = "MERGED2018_19_PP.csv"
    df_parent = pd.read_csv(filename_01)
    df_child = pd.read_csv(filename_02, low_memory=False)
    df_parent["UNITID"] = df_parent["UNITID"].astype("Int64", errors='ignore')
    df_child["UNITID"] = df_child["UNITID"].astype("Int64", errors='ignore')
    df = df_parent.merge(
    df_child,
    how = "left",
    on = "UNITID",
    copy = False
    )
    print(f'dataframe shape: {df.shape}')
    return df
