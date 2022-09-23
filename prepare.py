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
from scipy.stats.mstats import winsorize

# !iterative imputer must follow this import sequence!
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


# data processing functions for dept. of education - college scorecard dataset
# ---------------------------------------------------------------- #

def avg_net_price(df):

    '''Function that creates a new 'average net price' column from 
    existing avg net public and private columns.
    
    This function takes in a dataframe and re-labels null values as 0 in order
    to add across the two avg net price observations.'''

    df['avg_net_price_public'] = df['avg_net_price_public'].fillna(0)

    df['avg_net_price_private'] = df['avg_net_price_private'].fillna(0)

    df['avg_net_price'] = df.avg_net_price_public + df.avg_net_price_private

    # return the dataframe
    return df



def categorize_major(column):

    '''Function that places all major titles from `major_name` within 
    more concise buckets/categories for dimensionality reduction.'''

    if column in ['Botany/Plant Biology.','Agricultural Engineering.','Applied Horticulture and Horticultural Business Services.','Agriculture/Veterinary Preparatory Programs.','Soil Sciences.','Agriculture, General.', 'Agriculture, Agriculture Operations, and Related Sciences, Other.', 'Agricultural Production Operations.', 'Agricultural and Domestic Animal Services.','Agricultural Public Services.','Agricultural Mechanization.','International Agriculture.','Agricultural and Food Products Processing.']:
        return "Agriculture"
    elif column in ['Forest Engineering.','Environmental Control Technologies/Technicians.','Archeology.','Geological/Geophysical Engineering.','Mining and Mineral Engineering.','Natural Resources and Conservation, Other.','Fishing and Fisheries Sciences and Management.','Sustainability Studies.','Historic Preservation and Conservation.','Surveying Engineering.','Ocean Engineering.','Geography and Cartography.','Wildlife and Wildlands Science and Management.','Natural Resources Management and Policy.','Geological and Earth Sciences/Geosciences.','Environmental/Environmental Health Engineering.','Natural Resources Conservation and Research.','Forestry.']:
        return "Environment and Natural Resources"
    elif column in ['Drafting/Design Engineering Technologies/Technicians.','Architecture and Related Services, Other.','Architectural Sciences and Technology.','Interior Architecture.','Architectural Engineering.','Architecture.','Environmental Design.']:
        return "Architecture"
    elif column in ['Demography and Population Studies.','Medieval and Renaissance Studies.','Classical and Ancient Studies.','Bilingual, Multilingual, and Multicultural Education.','Museology/Museum Studies.','Science, Technology and Society.','Urban Studies/Affairs.','Cultural Studies/Critical Theory and Analysis.','African Languages, Literatures, and Linguistics.', 'Turkic, Uralic-Altaic, Caucasian, and Central Asian Languages, Literatures, and Linguistics.']:
        return "Area, Ethnic, and Civilization Studies"
    elif column in ['Telecommunications Management.','Communication, Journalism, and Related Programs, Other.','Communication and Media Studies.','Public Relations, Advertising, and Applied Communication.']:
        return "Communications"
    elif column in ['Communications Technologies/Technicians and Support Services, Other.','Educational/Instructional Media Design.','Graphic Communications.','Communications Technology/Technician.','Audiovisual Communications Technologies/Technicians.','Radio, Television, and Digital Communication.']:
        return "Communication Technologies"
    elif column in ['Accounting and Computer Science.','Human Computer Interaction.','Data Processing.','Computational Science.','Computer Software and Media Applications.','Computer and Information Sciences and Support Services, Other.','Computer Engineering Technologies/Technicians.','Computer Systems Analysis.','Computer Systems Networking and Telecommunications.','Computer Programming.','Computer/Information Technology Administration and Management.','Computer Science.','Information Science/Studies.','Computer Engineering.','Computer and Information Sciences, General.','Management Information Systems and Services.']:
        return "Computer and Information Sciences"
    elif column in ['Personal and Culinary Services, Other.','Cosmetology and Related Personal Grooming Services.','Nutrition Sciences.','Culinary Arts and Related Services.']:
        return "Cosmetology Services and Culinary Arts"
    elif column in ['Basic Skills and Developmental/Remedial Education.','Curriculum and Instruction.','High School/Secondary Diploma Programs.','High School/Secondary Certificate Programs.','Social and Philosophical Foundations of Education.','Teaching Assistants/Aides.','Student Counseling and Personnel Services.','Educational Administration and Supervision.','Teaching English or French as a Second or Foreign Language.','Education, Other.','Educational Assessment, Evaluation, and Research.','Education, General.','Special Education and Teaching.','Teacher Education and Professional Development, Specific Levels and Methods.','Teacher Education and Professional Development, Specific Subject Areas.']:
        return "Education Administration and Teaching"
    elif column in ['Ceramic Sciences and Engineering.','Electromechanical Engineering.','Biochemical Engineering.','Engineering Chemistry.','Mechatronics, Robotics, and Automation Engineering.','Engineering Mechanics.','Engineering Physics.','Engineering-Related Fields.','Engineering Science.','Petroleum Engineering.','Metallurgical Engineering.','Engineering, Other.','Industrial Engineering.','Chemical Engineering.','Aerospace, Aeronautical and Astronautical Engineering.','Engineering, General.','Electrical, Electronics and Communications Engineering.','Civil Engineering.','Mechanical Engineering.','Materials Engineering']:
        return "Engineering"
    elif column in ['Civil Engineering Technologies/Technicians.','Engineering-Related Technologies.','Engineering Technologies/Technicians, Other.','Engineering Technology, General.','Electrical Engineering Technologies/Technicians.','Mechanical Engineering Related Technologies/Technicians.','Construction Engineering Technologies.']:
        return "Engineering Technologies"
    elif column in ['Iranian/Persian Languages, Literatures, and Linguistics.','Turkic, Uralic-Altaic, Caucasian, and Central Asian Languages, Literatures, and Linguistics.','African Languages, Literatures, and Linguistics.','Celtic Languages, Literatures, and Linguistics.','South Asian Languages, Literatures, and Linguistics.','Middle/Near Eastern and Semitic Languages, Literatures, and Linguistics.','American Sign Language.','Slavic, Baltic and Albanian Languages, Literatures, and Linguistics.','Foreign Languages, Literatures, and Linguistics, Other.','American Indian/Native American Languages, Literatures, and Linguistics.','East Asian Languages, Literatures, and Linguistics.','Germanic Languages, Literatures, and Linguistics.','Modern Greek Language and Literature.','Southeast Asian and Australasian/Pacific Languages, Literatures, and Linguistics.','Linguistic, Comparative, and Related Language Studies and Services.','Romance Languages, Literatures, and Linguistics.']:
        return "Linguistics and Foreign Languages"
    elif column in ['Work and Family Studies.','Family and Consumer Sciences/Human Sciences Business Services.','Family and Consumer Sciences/Human Sciences, General.','Family and Consumer Sciences/Human Sciences, Other.','Hospitality Administration/Management.','Family and Consumer Economics and Related Studies.']:
        return "Family and Consumer Sciences"
    elif column in ['Law.','Legal Professions and Studies, Other.','Legal Research and Advanced Professional Studies.','Legal Support Services.','Non-Professional General Legal Studies (Undergraduate).']:
        return "Law"
    elif column in ['Creative Writing.','Publishing.','English Language and Literature/Letters, Other.','Literature.','Classics and Classical Languages, Literatures, and Linguistics.','English Language and Literature, General.','Journalism.']:
        return "English Language, Literature, and Composition"
    elif column in ['Liberal Arts and Sciences, General Studies and Humanities.']:
        return "Liberal Arts and Humanities"
    elif column in ['Library Science and Administration.','Library Science, Other.']:
        return "Library Science"
    elif column in ['Neuroscience.','Nanotechnology.','Biology Technician/Biotechnology Laboratory Technician.','Veterinary Medicine.','Maritime Studies.','Marine Sciences.','Pharmacology and Toxicology.','Human Biology.','Veterinary Biomedical and Clinical Sciences.','Atmospheric Sciences and Meteorology.','Biomathematics, Bioinformatics, and Computational Biology.','Cell/Cellular Biology and Anatomical Sciences.','Biological and Physical Sciences.','Biochemistry, Biophysics and Molecular Biology.','Zoology/Animal Biology.','Veterinary/Animal Health Technologies/Technicians.','Microbiological Sciences and Immunology.','Foods, Nutrition, and Related Services.','Ecology, Evolution, Systematics, and Population Biology.','Neurobiology and Neurosciences.','Genetics.','Animal Sciences.','Plant Sciences.','Food Science and Technology.','Chemistry.','Biology, General.','Biomedical/Medical Engineering.']:
        return "Biology and Life Sciences"
    elif column in ['Mathematics and Statistics, Other.','Mathematics and Computer Science.','Physics and Astronomy.','Statistics.','Mathematics.','Physics.','Astronomy and Astrophysics.','Applied Mathematics.']:
        return "Mathematics and Statistics"
    elif column in ['Military Science and Operational Studies.','Military Technologies and Applied Sciences, Other.','Air Force ROTC, Air Science and Operations.','Army ROTC, Military Science and Operations.','Intelligence, Command Control and Information Operations.','Naval Architecture and Marine Engineering.','Military Systems and Maintenance Technology.','Military Applied Sciences.','Security Science and Technology.']:
        return "Military Technologies"
    elif column in ['International and Comparative Education.','Systems Science and Theory.','Intercultural/Multicultural and Diversity Studies.','International/Global Studies.','Multi-/Interdisciplinary Studies, General.','Multi/Interdisciplinary Studies, Other.','Area Studies.']:
        return "Interdisciplinary and Multi-Disciplinary Studies (General)"
    elif column in ['Parks, Recreation and Leisure Facilities Management.','Movement and Mind-Body Therapies and Education.','Leisure and Recreational Activities.','Housing and Human Environments.','Landscape Architecture.','Outdoor Education.','Parks, Recreation, Leisure, and Fitness Studies, Other.','Health and Physical Education/Fitness.','Parks, Recreation and Leisure Studies.']:
        return "Physical Fitness, Parks, Recreation, and Leisure"
    elif column in ['Philosophy and Religious Studies, Other.','Philosophy and Religious Studies, General.','Religious Education.','Philosophy.','Bioethics/Medical Ethics.','Religious/Sacred Music.']:
        return "Philosophy and Religious Studies"
    elif column in ['Theology and Religious Vocations, Other.','Theological and Ministerial Studies.','Missions/Missionary Studies and Missiology.','Religion/Religious Studies.','Bible/Biblical Studies.','Pastoral Counseling and Specialized Ministries.']:
        return "Theology and Religious Vocations"
    elif column in ['Somatic Bodywork and Related Therapeutic Services.','Energy and Biologically Based Therapies.','Physical Science Technologies/Technicians.','Physiology, Pathology and Related Sciences.','Natural Sciences.','Physical Sciences.','Physical Sciences, Other.']:
        return "Physical Sciences"
    elif column in ['Nuclear and Industrial Radiologic Technologies/Technicians.','Nuclear Engineering.','Nuclear Engineering Technologies/Technicians.','Science Technologies/Technicians, Other.','Electromechanical Instrumentation and Maintenance Technologies/Technicians.']:
        return "Nuclear, Industrial Radiology, and Biological Technologies"
    elif column in ['Social Psychology.','Interpersonal and Social Skills.','Cognitive Science.','Biopsychology.','Research and Experimental Psychology.','Psychology, Other.','Clinical, Counseling and Applied Psychology.','Behavioral Sciences.','Clinical Psychology.','Human Development, Family Studies, and Related Services.']:
        return "Psychology"
    elif column in ['Homeland Security.','Homeland Security, Law Enforcement, Firefighting and Related Protective Services, Other.','International Relations and National Security Studies.','Fire Protection.','Criminal Justice and Corrections.','Criminology.']:
        return "Criminal Justice and Fire Protection"
    elif column in ['Security Policy and Strategy.','Taxation.','Citizenship Activities.','Peace Studies and Conflict Resolution.','Human Services, General.','Community Organization and Advocacy.','Mental and Social Health Services and Allied Professions.','Public Policy Analysis.','Public Administration and Social Service Professions, Other.','Public Administration.','Economics.','Rehabilitation and Therapeutic Professions.','City/Urban, Community and Regional Planning.','Social Work.','Political Science and Government.']:
        return "Public Affairs, Policy, and Social Work"
    elif column in ['Dispute Resolution.','Sociology and Anthropology.','Rural Sociology.','Social Sciences, General.','Communication Disorders Sciences and Services.','Human Development, Family Studies, and Related Services.','Sociology.','Psychology, General.','Ethnic, Cultural Minority, Gender, and Group Studies.','Anthropology.','Social Sciences, Other.']:
        return "Social Sciences"
    elif column in ['Carpenters.','Mason/Masonry.','Construction Trades, Other.','Construction Trades, General.','Woodworking.','Electrical and Power Transmission Installers.','Construction Management.','Building/Construction Finishing, Management, and Inspection.','Architectural Engineering Technologies/Technicians.','Heating, Air Conditioning, Ventilation and Refrigeration Maintenance Technology/Technician (HAC, HACR, HVAC, HVACR).','Construction Engineering.']:
        return "Construction Services"
    elif column in ['Heavy/Industrial Equipment Maintenance Technologies.','Vehicle Maintenance and Repair Technologies.','Electrical/Electronics Maintenance and Repair Technology.','Science Technologies/Technicians, General.','Energy Systems Technologies/Technicians.']:
        return "Electrical and Mechanic Repairs and Technologies"
    elif column in ['Paper Science and Engineering.','Precision Metal Working.','Materials Sciences.','Systems Engineering.','Manufacturing Engineering.','Quality Control and Safety Technologies/Technicians.','Industrial Production Technologies/Technicians.','Polymer/Plastics Engineering.','Apparel and Textiles.','Textile Sciences and Engineering.']:
        return "Precision Production and Industrial Arts"
    elif column in ['Mining and Petroleum Technologies/Technicians.','Marine Transportation.','Air Transportation.','Transportation and Materials Moving, Other.']:
        return "Transportation Sciences and Technologies"
    elif column in ['Crafts/Craft Design, Folk Art and Artisanry.','Visual and Performing Arts, Other.','Film/Video and Photographic Arts.','Visual and Performing Arts, General.','Design and Applied Arts.','Dance.','Rhetoric and Composition/Writing Studies.','Fine and Studio Arts.','Music.','Drama/Theatre Arts and Stagecraft.']:
        return "Fine Arts"
    elif column in ['Medical Clinical Sciences/Graduate Medical Studies.','Dentistry.','Alternative and Complementary Medical Support Services.','Optometry.','Health-Related Knowledge and Skills.','Funeral Service and Mortuary Science.','Gerontology.','Ophthalmic and Optometric Support Services and Allied Professions.','Alternative and Complementary Medicine and Medical Systems.','Chiropractic.','Podiatric Medicine/Podiatry.','Advanced/Graduate Dentistry and Oral Sciences.','Alternative and Complementary Medicine and Medical Systems.''Biological and Biomedical Sciences, Other.','Practical Nursing, Vocational Nursing and Nursing Assistants.','Pharmacy, Pharmaceutical Sciences, and Administration.','Medicine.','Medical Illustration and Informatics.','Allied Health and Medical Assisting Services.','Dental Support Services and Allied Professions.','Health/Medical Preparatory Programs.','Biological/Biosystems Engineering.','Biotechnology.','Nursing.','Health Professions and Related Clinical Sciences, Other.','Dietetics and Clinical Nutrition Services.','Registered Nursing, Nursing Administration, Nursing Research and Clinical Nursing.','Clinical/Medical Laboratory Science/Research and Allied Professions.','Public Health.','Health Services/Allied Health/Health Sciences, General.','Health and Medical Administrative Services.','Allied Health Diagnostic, Intervention, and Treatment Professions.']:
        return "Medical and Health Sciences and Services"
    elif column in ['Real Estate Development.','Operations Research.','Real Estate.','Insurance.','Specialized Sales, Merchandising and  Marketing Operations.','Arts, Entertainment,and Media Management.','Business Operations Support and Assistant Services.','Management Sciences and Quantitative Methods.','Business, Management, Marketing, and Related Support Services, Other.','Business/Commerce, General.','International Business.','Agricultural Business and Management.','Human Resources Management and Services.','General Sales, Merchandising and Related Marketing Operations.','Business/Managerial Economics.','Business/Corporate Communications.','Business Administration, Management and Operations.','Accounting and Related Services.','Entrepreneurial and Small Business Operations.','Finance and Financial Management Services.','Marketing.']:
        return "Business"
    elif column in ['History.','Holocaust and Related Studies.','Architectural History and Criticism.']:
        return "History"
    else:
        return "None"



# ---------- initial cleaning function ---------- #

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
            "STABBR": "state_post_code",
            "ZIP": "zip_code",
            "CITY": "city",
            "REGION": "region_ipeds",
            "OPEFLAG": "title_IV_eligibility",
            "PREDDEG": "pred_degree",
            "SCH_DEG": "pred_degree_0and4",
            "CREDDESC": "degree_name",
            "CREDLEV": "degree_code",
            "CIPCODE": "major_code",
            "CIPDESC": "major_name",
            "NUMBRANCH": "branch_number",
            "NPT4_PUB": "avg_net_price_public",
            "NPT4_PRIV": "avg_net_price_private",
            "NPT4_PROG": "avg_net_price_program",
            "NPT4_OTHER": "avg_net_price_other",
            'NUM41_PUB':'pub_fam_income_0_30000',
            'NUM41_PRIV':'private_fam_income_0_30000',
            'NUM41_PROG':'program_fam_income_0_30000',
            'NUM41_OTHER':'other_fam_income_0_30000',
            'NUM42_PUB':'pub_fam_income_30001_48000',
            'NUM42_PRIV':'private_fam_income_30001_48000',
            'NUM42_PROG':'program_fam_income_30001_48000',
            'NUM42_OTHER':'other_fam_income_30001_48000',
            'NUM43_PUB':'pub_fam_income_48001_75000',
            'NUM43_PRIV':'private_fam_income_48001_75000',
            'NUM43_PROG':'program_fam_income_48001_75000',
            'NUM43_OTHER':'other_fam_income_48001_75000',
            'NUM44_PUB':'pub_fam_income_75001_110000',
            'NUM44_PRIV':'private_fam_income_75001_110000',
            'NUM44_PROG':'program_fam_income_75001_110000',
            'NUM44_OTHER':'other_fam_income_75001_110000',
            'NUM45_PUB':'pub_fam_income_over_110000',
            'NUM45_PRIV':'private_fam_income_over_110000',
            'NUM45_PROG':'program_fam_income_over_110000',
            'NUM45_OTHER':'other_fam_income_over_110000',
            "NUM4_PRIV": "title_IV_student_number",
            "TUITFTE": "full_time_net_tuition_revenue",
            "ROOMBOARD_OFF": "off_campus_cost_of_attendace",
            "ROOMBOARD_ON": "on_campus_cost_of_attendace",
            "ADM_RATE": "admission_rate",
            "GRADS": "graduate_number",
            "ACTCMMID": "ACT_score_mid",
            "SAT_AVG": "avg_sat_admitted",
            "ADMCON7": "required_score",
            "AVGFACSAL": "avg_faculty_salary",
            "DISTANCEONLY": "online_only",
            "C150_4": "comp_rt_ft_150over_expected_time",
            "C150_4_2MOR": "comp_rt_ft_150over_expected_time_two_races",
            "C150_4_AIAN": "comp_rt_ft_150over_expected_time_native_american",
            "C150_4_ASIAN": "comp_rt_ft_150over_expected_time_asian",
            "C150_4_BLACK": "comp_rt_ft_150over_expected_time_black",
            "C150_4_HISP": "comp_rt_ft_150over_expected_time_hispanic",
            "C150_4_NRA": "comp_rt_ft_150over_expected_time_non_resident",
            "C150_4_UNKN": "comp_rt_ft_150over_expected_time_unknown_race",
            "C150_4_WHITE": "comp_rt_ft_150over_expected_time_white",
            "PFTFTUG1_EF": "share_entering_students_first_ft",
            "PPTUG_EF": "share_of_part_time",
            "RET_FT4": "first_time_ft_student_retention",
            "RET_PT4": "first_time_pt_student_retention",
            "UGDS_2MOR": "enrollment_share_two_races",
            "UGDS_AIAN": "enrollment_share_native_american",
            "UGDS_ASIAN": "enrollment_share_asian",
            "UGDS_BLACK": "enrollment_share_black",
            "UGDS_HISP": "enrollment_share_hispanic",
            "UGDS_NHPI": "enrollment_share_pac_islander",
            "UGDS_NRA": "enrollment_share_non_resident",
            "UGDS_UNKN": "enrollment_share_unknown",
            "UGDS_WHITE": "enrollment_share_white",
            "D_PCTPELL_PCTFLOAN": "undergraduate_number_pell_grant_fedral_loan",
            "DEBT_MDN": "median_loan_repayment",
            "PELL_DEBT_MDN": "med_debt_pell_students",
            "LO_INC_DEBT_MDN": "median_debt_0_30000",
            "MD_INC_DEBT_MDN": "median_debt_30001_75000",
            "HI_INC_DEBT_MDN": "median_debt_75001+",
            "GRAD_DEBT_MDN": "median_debt_completed",
            "WDRAW_DEBT_MDN": "not_completed_med_debt",
            "MALE_DEBT_MDN": "median_debt_male",
            "FEMALE_DEBT_MDN": "median_debt_female",
            "IND_DEBT_MDN": "median_debt_independent",
            "FIRSTGEN_DEBT_MDN": "median_debt_first_generation",
            "NOTFIRSTGEN_DEBT_MDN": "median_debt_non_first_generation",
            "NOPELL_DEBT_MDN": "median_debt_non_pell",
            "FTFTPCTFLOAN": "fedral_loan_full_time_first_time_undergraduate",
            "FTFTPCTPELL": "pell_grant_full_time_first_time_undergraduate",
            "DEBT_PELL_PP_EVAL_MDN": "med_parent_and_pell",
            "DEBT_PELL_PP_EVAL_MEAN": "avg_parent_and_pell",
            "DEBT_PELL_STGP_EVAL_MDN": "med_stafford_and_pell",
            "DEBT_PELL_STGP_EVAL_MEAN": "avg_stafford_and_pell",
            "DEBT_ALL_PP_EVAL_MDN": "med_parent_and_loan",
            "DEBT_ALL_PP_EVAL_MEAN": "avg_parent_and_loan",
            "DEBT_ALL_STGP_EVAL_MDN": "med_stafford_and_debt",
            "DEBT_ALL_STGP_EVAL_MEAN": "avg_stafford_and_debt",
            "DEBT_ALL_STGP_EVAL_MDN10YRPAY": "med_stafford_and_grad_debt",
            "DEBT_NOPELL_STGP_EVAL_MDN": "med_stafford_and_no_pell_recipients",
            "DEBT_NOPELL_STGP_EVAL_MEAN": "avg_stafford_and_no_pell_recipients",
            "DEBT_ALL_PP_EVAL_MDN10YRPAY": "med_monthly_payment_parent_and_debt",
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
            "PCIP27": "deg_percent_awarded_mathematics",
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
            "UGNONDS": "non_deg_seeking"
        })

    # collapse average net price by institution control columns
    new_df = avg_net_price(new_df)

    # create the 'major_category' bin column
    new_df['major_category'] = new_df["major_name"].apply(categorize_major)

    return new_df


def fill_null_with_mean(df):

    '''Function to fill-in a by major average for entrance exams/admissions rate.'''
    
    df['admission_rate'] = df.groupby('major_category')['admission_rate'].apply(lambda x:x.fillna(x.mean()))
    df['ACT_score_mid'] = df.groupby('major_category')['ACT_score_mid'].apply(lambda x:x.fillna(x.mean()))
    df['avg_sat_admitted'] = df.groupby('major_category')['avg_sat_admitted'].apply(lambda x:x.fillna(x.mean()))
    return df


def clean_high_percentage_nulls(df):

    '''Function takes in the initial df and 
    cleans for high percentage null value features (>40%).'''

    df = df.dropna(subset=['city'])

    new_df = df.drop(columns = [ 
        'avg_net_price_other',    
        'avg_net_price_program',     
        'med_parent_and_pell',    
        'avg_parent_and_pell',    
        'med_monthly_payment_parent_and_debt',
        'med_parent_and_loan',    
        'avg_parent_and_loan',   
        'avg_stafford_and_no_pell_recipients',
        'avg_stafford_and_pell',   
        'med_stafford_and_pell',     
        'med_stafford_and_no_pell_recipients',
        'avg_stafford_and_debt',     
        'med_stafford_and_debt',     
        'med_stafford_and_grad_debt',
        'title_IV_student_number',
        'median_loan_repayment',                       
        'median_debt_0_30000',              
        'median_debt_30001_75000',          
        'median_debt_75001+',                         
        'not_completed_med_debt',           
        'median_debt_male',                 
        'median_debt_female',               
        'median_debt_independent',          
        'median_debt_first_generation'      
    ])

    # fill admission test score nulls with averages
    new_df = fill_null_with_mean(new_df)

    # print the new df
    print(f'dataframe shape: {new_df.shape}')

    # return the new df
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

    print(f'modified dataframe shape: {mod_df.shape}')
    
    # return the new df
    return mod_df


#### ------------------------------------- ####
### Additional Cleaning Functions ### 


## Code to apply above function to our df, creating new `major_category` column/feature
# --> new_df['major_category'] = new_df.major_name.apply(categorize_major)
# ----------------------------------- #

def obtain_target_variables(df):

    '''Function to perform merge with `earnings_pivot_merge` df'''

    # Reading in csv of earnings pivot table (creation of Chenchen)
    earnings_pivot_merge = pd.read_csv('2017_2018_2019_earning_by_major.csv', index_col=0)

    # Merging cleaned/prepared df with earnings pivot table
    df = df.merge(earnings_pivot_merge, how='inner', on='major_category')

    # get target variables
    new_df = create_roi_cols(df).round(4)

    # print dataframe shape
    print(f'dataframe shape: {new_df.shape}')

    # return the newly transformed dataframe
    return new_df


# ----------------------------------- #

### Target Variable: ROI ###

''' These features intake calculated median earnings data from our secondary IPUMS dataset by year (`median_earnings_by_degree`), 
net college cost of a typical 4-yr bachelors degree, and predicted counter earnings had an individual not pursued this degree. 
It utilizes a standard ROI formula calculation to engineer new ROI vars for 5, 10, and 20 years.
This is our primary target variable'''

# 5-yr ROI 

def roi_5yr(df):

    # creating median earnings var
    median_earnings_by_degree_5yr = (df['2017'] + df['2018'] + df['2019'] + df['2019']*1.02 + (df['2019']*1.02)*1.02)

    # net college cost var
    net_college_cost = df['avg_net_price']*4

    # counter earnings var (what is the predicted wage an individual would have earned had they foregone pursuing this degree)
    counter_earnings = (39070*4)

    # 
    net_cost_of_investment = (net_college_cost + counter_earnings)

    #
    net_return_on_investment_5yr = median_earnings_by_degree_5yr - net_cost_of_investment


    # ROI formula calculation
    df['roi_5yr'] = net_return_on_investment_5yr / net_cost_of_investment


    # ROI calculation as a percentage
    df['pct_roi_5yr'] = (net_return_on_investment_5yr / net_cost_of_investment) * 100

    return df


# 10-yr ROI

def roi_10yr(df):

    # creating median earnings var
    median_earnings_by_degree_10yr = df['2017'] + df['2018'] + df['2019'] + df['2019']*1.02 + (df['2019']*1.02)*1.02 + ((df['2019']*1.02)*1.02)*1.02 + (((df['2019']*1.02)*1.02)*1.02)*1.02 + ((((df['2019']*1.02)*1.02)*1.02)*1.02)*1.02 + (((((df['2019']*1.02)*1.02)*1.02)*1.02)*1.02)*1.02 + ((((((df['2019']*1.02)*1.02)*1.02)*1.02)*1.02)*1.02)*1.02

    # net college cost var
    net_college_cost = df['avg_net_price']*4

    # counter earnings var (what is the predicted wage an individual would have earned had they foregone pursuing this degree)
    counter_earnings = (39070*4)

    # 
    net_cost_of_investment = (net_college_cost + counter_earnings)

    #
    net_return_on_investment_10yr = median_earnings_by_degree_10yr - net_cost_of_investment


    # ROI formula calculation
    df['roi_10yr'] = net_return_on_investment_10yr / net_cost_of_investment

    # ROI calculation as a percentage
    df['pct_roi_10yr'] = (net_return_on_investment_10yr / net_cost_of_investment) * 100

    return df



# 20-yr ROI

def roi_20yr(df):

    # creating median earnings var                                                                                                                                                                                          yr10                   yr11                                                                                     yr15
    median_earnings_by_degree_20yr = df['2017'] + df['2018'] + df['2019'] + df['2019']*1.02 + df['2019']*(1.02**2) + df['2019']*(1.02**3) + df['2019']*(1.02**4) + df['2019']*(1.02**5) + df['2019']*(1.02**6) + df['2019']*(1.02**7) + df['2019']*(1.02**8) + df['2019']*(1.02**9) + df['2019']*(1.02**10) + df['2019']*(1.02**11) + df['2019']*(1.02**12) + df['2019']*(1.02**13) + df['2019']*(1.02**14) + df['2019']*(1.02**15) + df['2019']*(1.02**16) + df['2019']*(1.02**17)

    # net college cost var
    net_college_cost = df['avg_net_price']*4

    # counter earnings var (what is the predicted wage an individual would have earned had they foregone pursuing this degree)
    counter_earnings = (39070*4)

    # 
    net_cost_of_investment = (net_college_cost + counter_earnings)

    #
    net_return_on_investment_20yr = median_earnings_by_degree_20yr - net_cost_of_investment


    # ROI formula calculation
    df['roi_20yr'] = net_return_on_investment_20yr / net_cost_of_investment

    # ROI calculation as a percentage
    df['pct_roi_20yr'] = (net_return_on_investment_20yr / net_cost_of_investment) * 100

    return df


### Master function for all roi vars ###
def create_roi_cols(df):

    # Calling roi_5yr function
    df = roi_5yr(df)

    # Calling roi_10yr function
    df = roi_10yr(df)

    # Calling roi_10yr function
    df = roi_20yr(df)

    return df


# ----------------------------------- #

# create a "collapse" cols function

def get_fam_income_col(df, col_lst, new_col_string):

    '''Function that creates a new family income columns from 
    existing dummy columns.'''

    df[col_lst] = df[col_lst].fillna(0)

    df[new_col_string] = df[col_lst].sum(axis = 1)

    # drop redundant columns
    df = df.drop(df[col_lst], axis = 1)

    # return the dataframe
    return df

def create_fam_income_columns(df):
    
    # income brackets
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

    # list of cols to collapse
    frames = [
        income_0_30000,
        income_30001_48000, 
        income_48001_75000, 
        income_75001_110000, 
        income_over_110000]

    # list of new col names
    var_names = [
        'income_0_30000',
        'income_30001_48000', 
        'income_48001_75000', 
        'income_75001_110000', 
        'income_over_110000']

    for i in range(len(frames)):
        var_name = var_names[i]
        df = get_fam_income_col(df, frames[i], var_name)

    print(f'dataframe shape: {df.shape}')
    
    return df



# ---------------------------------------------------------------- #
                    ### Train, Validate, Test Split ###
# ---------------------------------------------------------------- #

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
    
    print(f'train shape: {train.shape}')
    print(f'validate shape: {validate.shape}')
    print(f'test shape: {test.shape}')

    return train, validate, test

# Handles outliers 
def percentile_capping(df, low_end = 0.1, high_end = 0.1):

    '''Function that uses scipy's winsorize method to cap
    continuous variables at lower and higher end based on a passed 
    percentile values.'''

    l1 = df.select_dtypes(include = "number").columns.tolist()

    # dont include target variables to cap
    target_lst = [ 
                "roi_5yr",
                "roi_10yr",
                "roi_20yr",
                "pct_roi_5yr",
                "pct_roi_10yr",
                "pct_roi_20yr"
                "2017",                                               
                "2018",                                                   
                "2019",
                "Grand Total",
                "avg_net_price",
                "med_debt_pell_students",
                "median_debt_non_pell",
                "median_debt_completed"]

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
    '''using sklearn's iterative imputer to fill-in remaining nulls. Placeholder for continuous features.'''

    l1 = train_df.select_dtypes(include = "number").columns.tolist()

    # dont learn from these variables
    target_lst = [ 
        "roi_5yr",
        "roi_10yr",
        "roi_20yr",
        "pct_roi_5yr",
        "pct_roi_10yr",
        "pct_roi_20yr"
        "2017",                                               
        "2018",                                                   
        "2019",
        "Grand Total",
        "avg_net_price",
        "med_debt_pell_students",
        "median_debt_non_pell",
        "median_debt_completed"
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

    # fill-in missing zip codes
    train_df["zip_code"] = train_df["zip_code"].fillna(train_df["zip_code"].mode()[0])

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
        "roi_20yr",
        "pct_roi_5yr",
        "pct_roi_10yr",
        "pct_roi_20yr"
        "2017",                                               
        "2018",                                                   
        "2019",
        "Grand Total",
        "avg_net_price",
        "med_debt_pell_students",
        "median_debt_non_pell",
        "median_debt_completed"
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


def fill_null_with_mean(df):
    '''fill the null value with avg for each major'''
    
    df['admission_rate'] = df.groupby('major_category')['admission_rate'].apply(lambda x:x.fillna(x.mean()))
    df['ACT_score_mid'] = df.groupby('major_category')['ACT_score_mid'].apply(lambda x:x.fillna(x.mean()))
    df['avg_sat_admitted'] = df.groupby('major_category')['avg_sat_admitted'].apply(lambda x:x.fillna(x.mean()))
    
    return df

def impute_avg_net_price(df):
    # Impute `avg_net_price` where value = 0; perform split by `institution_control` var (Public, Private For-Profit, Private Non-Profit)

    df.loc[(df.institution_control == 'Public') & (df.avg_net_price == 0), 'avg_net_price'] = 14502
    df.loc[(df.institution_control == 'Private, nonprofit') & (df.avg_net_price == 0), 'avg_net_price'] = 22961
    df.loc[(df.institution_control == 'Private, for-profit') & (df.avg_net_price == 0), 'avg_net_price'] = 18640

    return df

def impute_debt(df):

    df['med_debt_pell_students'] = np.where(df['med_debt_pell_students'].isna(), 17500, df['med_debt_pell_students'])
    df['median_debt_non_pell'] = np.where(df['median_debt_non_pell'].isna(), 14768, df['median_debt_non_pell'])
    df['median_debt_completed'] = np.where(df['median_debt_completed'].isna(), 23250, df['median_debt_completed'])

    df = df.astype({'med_debt_pell_students': 'int64', 'median_debt_non_pell': 'int64', 'median_debt_completed': 'int64'})
    
    return df


def manual_imputer(df):
    # calls all manual imputation functions for important vars
    df = fill_null_with_mean(df)
    df = impute_avg_net_price(df)
    df = impute_debt(df)

    return df










