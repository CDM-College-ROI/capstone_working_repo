# notebook dependencies 
import pandas as pd
import numpy as np
import os

# visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# initial acquisition functions
# -------------------------------------------- #

def get_majors_df():

    '''Function to initially pull and merge the two (2) needed 
    College Scorecard tables for period 2018-2019.'''

    # checking if dataset exists
    filename = "majors_table.csv"
    
    if os.path.isfile(filename):
        
        df = pd.read_csv(filename, index_col = True)

        print(f"dataframe shape: {df.shape}")

        return df

    else:
        # checks local foldere for following files
        filename_01 = "FieldOfStudyData1718_1819_PP.csv"
        filename_02 = "MERGED2018_19_PP.csv"
        
        # created the necessary parent and child tables
        df_parent = pd.read_csv(filename_01, low_memory=False)
        df_child = pd.read_csv(filename_02, low_memory=False)

        df_parent["UNITID"] = df_parent["UNITID"].astype("Int32", errors="ignore")
        df_child["UNITID"] = df_child["UNITID"].astype("Int32", errors="ignore")

        df = df_parent.merge( 
        df_child,
        how = "left",
        on = "UNITID",
        copy = False
        )
        # cache the newly created dataframe as a .csv file
        df.to_csv("majors_table.csv")
        # print the df shape
        print(f"dataframe shape: {df.shape}")

        # return the dataframe
        return df


def get_bach_df():

    '''Function to initial check for a bachelor degree table.
    
    If the table is not found, then it checks for the initially needed 
    College Scorecard tables for period 2018-2019.
    
    The function then filters and returns bachelor degree records.'''

    # checking if dataset exists
    filename = "bach_table.csv"
    
    if os.path.isfile(filename):
        
        df = pd.read_csv(filename, index_col = 0)

        print(f"dataframe shape: {df.shape}")

        return df

    else:
        # checks local foldere for following files
        filename_01 = "FieldOfStudyData1718_1819_PP.csv"
        filename_02 = "MERGED2018_19_PP.csv"
        
        # created the necessary parent and child tables
        df_parent = pd.read_csv(filename_01, low_memory=False)
        df_child = pd.read_csv(filename_02, low_memory=False)

        df_parent["UNITID"] = df_parent["UNITID"].astype("Int32", errors="ignore")
        df_child["UNITID"] = df_child["UNITID"].astype("Int32", errors="ignore")

        df = df_parent.merge( 
        df_child,
        how = "left",
        on = "UNITID",
        copy = False
        )

        # filters for just bachelor specific records
        bach_df = df[df["CREDDESC"] == "Bachelors Degree"]

        # initial filter of columns with >= 50% missing records
        bach_df = bach_df[[ 
                "UNITID",
                "INSTNM_x",
                "CONTROL_x",
                "CITY",
                "STABBR",
                "ZIP",
                "PFTFTUG1_EF",
                "PPTUG_EF",
                "PREDDEG",
                "REGION",
                "RET_FT4",
                "RET_PT4",
                "ROOMBOARD_OFF",
                "ROOMBOARD_ON",
                "SAT_AVG",
                "SCH_DEG",
                "TUITFTE",
                "ACTCMMID",
                "ADM_RATE",
                "ADMCON7",
                "AVGFACSAL",
                "CREDDESC",
                "CREDLEV",
                "CIPCODE",
                "CIPDESC",
                "DISTANCEONLY",
                "GRADS",
                "NUM4_PRIV",
                "NUMBRANCH",
                "NPT4_PRIV",
                "NPT4_PUB",
                "OPEFLAG",
                "NUM41_OTHER",
                "NUM41_PRIV",
                "NUM41_PROG",
                "NUM41_PUB",
                "NUM42_OTHER",
                "NUM42_PRIV",
                "NUM42_PROG",
                "NUM42_PUB",
                "NUM43_OTHER",
                "NUM43_PRIV",
                "NUM43_PROG",
                "NUM43_PUB",
                "NUM44_OTHER",
                "NUM44_PRIV",
                "NUM44_PROG",
                "NUM44_PUB",
                "NUM45_OTHER",
                "NUM45_PRIV",
                "NUM45_PROG",
                "NUM45_PUB",
                "PCIP01",
                "PCIP03",
                "PCIP04",
                "PCIP05",
                "PCIP09",
                "PCIP10",
                "PCIP11",
                "PCIP12",
                "PCIP13",
                "PCIP14",
                "PCIP15",
                "PCIP16",
                "PCIP19",
                "PCIP22",
                "PCIP23",
                "PCIP24",
                "PCIP25",
                "PCIP26",
                "PCIP27",
                "PCIP29",
                "PCIP30",
                "PCIP31",
                "PCIP38",
                "PCIP39",
                "PCIP40",
                "PCIP41",
                "PCIP42",
                "PCIP43",
                "PCIP44",
                "PCIP45",
                "PCIP46",
                "PCIP47",
                "PCIP48",
                "PCIP49",
                "PCIP50",
                "PCIP51",
                "PCIP52",
                "PCIP54",
                "C150_4_2MOR",
                "C150_4_AIAN",
                "C150_4_ASIAN",
                "C150_4_BLACK",
                "C150_4_HISP",
                "C150_4_NRA",
                "C150_4_UNKN",
                "C150_4_WHITE",
                "C150_4",
                "UGDS_2MOR",
                "UGDS_AIAN",
                "UGDS_ASIAN",
                "UGDS_BLACK",
                "UGDS_HISP",
                "UGDS_NHPI",
                "UGDS_NRA",
                "UGDS_UNKN",
                "UGDS_WHITE",
                "UGNONDS",
                "D_PCTPELL_PCTFLOAN",
                "DEBT_MDN",
                "PELL_DEBT_MDN",
                "WDRAW_DEBT_MDN",
                "FEMALE_DEBT_MDN",
                "FIRSTGEN_DEBT_MDN",
                "IND_DEBT_MDN",
                "LO_INC_DEBT_MDN",
                "MALE_DEBT_MDN",
                "MD_INC_DEBT_MDN",
                "NOPELL_DEBT_MDN",
                "NOTFIRSTGEN_DEBT_MDN",
                "HI_INC_DEBT_MDN",
                "GRAD_DEBT_MDN",
                "FTFTPCTFLOAN",
                "FTFTPCTPELL"]]
            
        # cache the newly created dataframe as a .csv file
        bach_df.to_csv("bach_table.csv")
        
        # print the df shape
        print(f"dataframe shape: {bach_df.shape}")

        # return the dataframe
        return bach_df

