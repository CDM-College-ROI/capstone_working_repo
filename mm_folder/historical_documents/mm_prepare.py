# notebook dependencies 
import pandas as pd
import numpy as np
import os

# visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# data processing functions for dept. of education - college scorecard dataset
# ---------------------------------------------------------------- #


def clean_college_df(df):
    '''Function to clean the intiail bachelor dataframe. This function takes 
    in the bachelor dataframe and checks for instances of 'PrivacySuppressed' 
    entries across the dataframe and replaces them with np.NaN.

    function also renames/cleans columns names for easier readability.'''

    # rename omitted entry values
    new_df = df.apply(lambda x: x.replace({'PrivacySuppressed': np.NaN}, regex=True))

    new_df = new_df.rename(columns = { 
            "UNITID": "unit_id_institution",
            "INSTNM_x": "college_name",
            "CONTROL_x": "institution_control",
            "CITY": "city",
            "STABBR": "state_post_code",
            "ZIP": "zip_code",
            "PFTFTUG1_EF": "share_entering_students_first_ft",
            "PPTUG_EF": "share_of_part_time",
            "PREDDEG": "pred_degree",
            "REGION": "religion_ipeds",
            "RET_FT4": "first_time_ft_student_retention",
            "RET_PT4": "first_time_pt_student_retention",
            "ROOMBOARD_OFF": "off_campus_cost_of_attendace",
            "ROOMBOARD_ON": "on_campus_cost_of_attendace",
            "SAT_AVG": "avg_sat_admitted",
            "SCH_DEG": "pred_degree_0and4",
            "TUITFTE": "full_time_net_tuition_revenue",
            "ACTCMMID": "ACT_score_mid",
            "ADM_RATE": "admission_rate",
            "ADMCON7": "required_score",
            "AVGFACSAL": "avg_faculty_salary",
            "CREDDESC": "degree_name",
            "CREDLEV": "degree_code",
            "CIPCODE": "major_code",
            "CIPDESC": "major_name",
            "DISTANCEONLY": "online_only",
            "GRADS": "graduate_number",
            "NUM4_PRIV": "title_IV_student_number",
            "NUMBRANCH": "branch_number",
            "NPT4_PRIV": "avg_net_price_public",
            "NPT4_PUB": "avg_net_price_private",
            "OPEFLAG": "title_IV_eligibility",
            "NUM41_OTHER": "other_fam_income_0_30000",
            "NUM41_PRIV": "private_fam_income_0_30000",
            "NUM41_PROG": "program_fam_income_0_30000",
            "NUM41_PUB": "pub_fam_income_0_30000",
            "NUM42_OTHER": "other_fam_income_30001_48000",
            "NUM42_PRIV": "private_fam_income_30001_48000",
            "NUM42_PROG": "program_fam_income_30001_48000",
            "NUM42_PUB": "pub_fam_income_30001_48000",
            "NUM43_OTHER": "other_fam_income_48001_75000",
            "NUM43_PRIV": "private_fam_income_48001_75000",
            "NUM43_PROG": "program_fam_income_48001_75000",
            "NUM43_PUB": "pub_fam_income_48001_75000",
            "NUM44_OTHER": "other_fam_income_75001_110000",
            "NUM44_PRIV": "private_fam_income_75001_110000",
            "NUM44_PROG": "program_fam_income_75001_110000",
            "NUM44_PUB": "pub_fam_income_75001_110000",
            "NUM45_OTHER": "other_fam_income_over_110000",
            "NUM45_PRIV": "private_fam_income_over_110000",
            "NUM45_PROG": "program_fam_income_over_110000",
            "NUM45_PUB": "pub_fam_income_over_110000",
            "PCIP01": "deg_percent_awarded_agriculture_operations",
            "PCIP03": "deg_percent_awarded_natural_resources",
            "PCIP04": "deg_percent_awarded_architecture",
            "PCIP05": "deg_percent_awarded_area_ethnic_cultural_gender",
            "PCIP09": "deg_percent_awarded_communication_journalism",
            "PCIP10": "deg_percent_awarded_communication_tech",
            "PCIP11": "deg_percent_awarded_computer_science",
            "PCIP12": "deg_percent_awarded_personal_culinary_services",
            "PCIP13": "deg_percent_awarded_education",
            "PCIP14": "deg_percent_awarded_engineering",
            "PCIP15": "deg_percent_awarded_engineering_tech",
            "PCIP16": "deg_percent_awarded_foreign_language_literatures",
            "PCIP19": "deg_percent_awarded_human_science",
            "PCIP22": "deg_percent_awarded_legal_profession",
            "PCIP23": "deg_percent_awarded_english_lang",
            "PCIP24": "deg_percent_awarded_general_studies",
            "PCIP25": "deg_percent_awarded_library_sciences",
            "PCIP26": "deg_percent_awarded_bio_sciences",
            "PCIP27": "deg_percent_awarded_history",
            "PCIP29": "deg_percent_awarded_military_tech",
            "PCIP30": "deg_percent_awarded_intedisciplinary_studies",
            "PCIP31": "deg_percent_awarded_leisure_fitness",
            "PCIP38": "deg_percent_awarded_philosophy",
            "PCIP39": "deg_percent_awarded_theology",
            "PCIP40": "deg_percent_awarded_physical_sciences",
            "PCIP41": "deg_percent_awarded_science_tech",
            "PCIP42": "deg_percent_awarded_psychology",
            "PCIP43": "deg_percent_awarded_homeland_security",
            "PCIP44": "deg_percent_awarded_public_admin",
            "PCIP45": "deg_percent_awarded_social_sciences",
            "PCIP46": "deg_percent_awarded_construction_trades",
            "PCIP47": "deg_percent_awarded_mechanic_repair",
            "PCIP48": "deg_percent_awarded_precision_production",
            "PCIP49": "deg_percent_awarded_transportation_materials",
            "PCIP50": "deg_percent_awarded_visual_and_performing_arts",
            "PCIP51": "deg_percent_awarded_health",
            "PCIP52": "deg_percent_awarded_business_management",
            "PCIP54": "deg_percent_awarded_history",
            "C150_4_2MOR": "comp_rt_ft_150over_expected_time_two_races",
            "C150_4_AIAN": "comp_rt_ft_150over_expected_time_native_american",
            "C150_4_ASIAN": "comp_rt_ft_150over_expected_time_asian",
            "C150_4_BLACK": "comp_rt_ft_150over_expected_time_black",
            "C150_4_HISP": "comp_rt_ft_150over_expected_time_hispanic",
            "C150_4_NRA": "comp_rt_ft_150over_expected_time_non_resident",
            "C150_4_UNKN": "comp_rt_ft_150over_expected_time_unknown_race",
            "C150_4_WHITE": "comp_rt_ft_150over_expected_time_white",
            "C150_4": "comp_rt_ft_150over_expected_time",
            "UGDS_2MOR": "enrollment_share_two_races",
            "UGDS_AIAN": "enrollment_share_native_american",
            "UGDS_ASIAN": "enrollment_share_asian",
            "UGDS_BLACK": "enrollment_share_black",
            "UGDS_HISP": "enrollment_share_hispanic",
            "UGDS_NHPI": "enrollment_share_pac_islander",
            "UGDS_NRA": "enrollment_share_non_resident",
            "UGDS_UNKN": "enrollment_share_unknown",
            "UGDS_WHITE": "enrollment_share_white",
            "UGNONDS": "non_deg_seeking",
            "D_PCTPELL_PCTFLOAN": "undergraduate_number_pell_grant_fedral_loan",
            "DEBT_MDN": "median_loan_repayment",
            "PELL_DEBT_MDN": "med_debt_pell_students",
            "WDRAW_DEBT_MDN": "not_completed_med_debt",
            "FEMALE_DEBT_MDN": "median_debt_female",
            "FIRSTGEN_DEBT_MDN": "median_debt_first_generation",
            "IND_DEBT_MDN": "median_debt_independent",
            "LO_INC_DEBT_MDN": "median_debt_0_30000",
            "MALE_DEBT_MDN": "median_debt_male",
            "MD_INC_DEBT_MDN": "median_debt_30001_75000",
            "NOPELL_DEBT_MDN": "median_debt_non_pell",
            "NOTFIRSTGEN_DEBT_MDN": "median_debt_non_first_generation",
            "HI_INC_DEBT_MDN": "median_debt_75001+",
            "GRAD_DEBT_MDN": "median_debt_completed",
            "FTFTPCTFLOAN": "fedral_loan_full_time_first_time_undergraduate",
            "FTFTPCTPELL": "pell_grant_full_time_first_time_undergraduate"
    })

    return new_df


def nulls_by_col(df):
    '''Function to return percentage of missing values by feature.'''

    num_missing = df.isnull().sum()

    rows = df.shape[0]

    prcnt_miss = num_missing / rows * 100

    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 'percent_rows_missing': prcnt_miss})
    
    return cols_missing.sort_values(by='num_rows_missing', ascending=False)


def treat_bach_nulls(df):
    '''Function to comb through the df for cols with greater than 50% missing values, 
    which are then dropped from the df and returned as a new df.'''

    # drop columns containing either 50% or more than 50% NaN Values
    perc = 50.0
    min_count =  int(((100-perc)/100) * df.shape[0] + 1)
    mod_df = df.dropna(
                axis=1, 
                thresh = min_count)

    print(f'modified df shape: {mod_df.shape}')
    
    # return the new df
    return mod_df