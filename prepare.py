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

# ---------------------------------------------------------------- #


def categorize_major(column):
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
