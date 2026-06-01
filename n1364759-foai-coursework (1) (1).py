#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries and loading dataset for the model

# In[1]:


#We are implementing the libraries below written

#reading the csv file required for model to run


# In[2]:


import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

df = pd.read_csv('/kaggle/input/newmodel/coursewoork1.csv')


# In[3]:


# Output of first five rows is printed through Head function


# In[4]:


df.head()


# ## Renaming the columns for simpler understanding of the code

# In[5]:


# Using the Rename function each row has been modified 


# In[6]:


df = df.rename(
    columns={
        "How old are you?": "age",
        "What industry is your employer in?": "industry",
        "What is the functional area of your job (this might be different from your company's industry)?": "job_function",
        "If your job title needs additional context, please clarify here:": "position_details",
        "What is your annual salary? This should be your GROSS (pre-tax) income. (You'll indicate the currency in a later question.) If you are part-time or hourly, please enter an annualized equivalent -- what you would earn if you worked the job 40 hours a week, 52 weeks a year.": "salary",
        "How much additional monetary compensation do you get, if any (for example, bonuses or overtime in an average year)? Only include monetary compensation here, not the value of benefits, tuition reimbursement, etc. If your bonus or overtime varies from year to year, use the most recent figures.": "rewards",
        "Please indicate the currency": "currency",
        'If "Other," please indicate the currency here: ' : 'foreign_currency',
        "If your income needs additional context, please provide it here:": "income_status",
        "What country do you work in? (Countries listed had by far the largest representation last year. Please write in your country if it's not listed.)": "workplace",
        "If you're in the U.S., what state do you work in?": "state_in_us",
        "What city/region do you work in?": "region",
        "Are you remote or on-site?" : "hybrid_on_site",
        "Is your job unionized?": "organised",
        "How many years of professional work experience do you have overall?": "track_record",
        "How many years of professional work experience do you have in your field?": "field_experience",
        "What is your highest level of education completed?": "qualification",
        "What is your gender?" : "gender",
        "What is your race? (Choose all that apply.)": "specify_race",
    }
)


# In[7]:


# Importing library "Warning" for Outcastng all the error messages arrives
#for better presentation of the output 

import warnings

warnings.filterwarnings("ignore")


# ### Checking the output with above library used via Head function.
# 

# In[8]:


df.head(5)


# ## Data Cleaning is the crucial Procedure required for model to predict the accuracy and we will be starting with cleaning the column Industry
# 
# # *Industry*

# In[9]:


#we will be printing total number of industry alloted with updated name and its count.


# In[10]:


print('number of industry:', df['industry'].unique().shape[0])


# In[11]:


#evaluating the data with top 12 industry and splitting it into smaller groups.

# By Value_counts we have obtained the output with 12 industries.


# In[12]:


df['industry'].value_counts().head(12)


# ### Segregating the above 12 industries into smaller groups so that our data gets cleaned and equally divided into smaller groups for better accuracy rate.
# 
# #### We are Defining the data into four new variables and by doing that all the values gets into that variables and the remaining data gets into the separate column named Others,that include rest of the data apart from the earlier variables name.

# In[13]:


Education_tech =['Education (Higher Education)','Education (Primary/Secondary)','Engineering','Computing or Tech','Aerospace & Defense']
Gov_manufacturing =['Government & Public Administration','Manufacturing','Galleries, Libraries, Archives & Museums']
Media_healthcare =['Health care','Utilities & Telecommunications','Media & Digital']
Marketing = ['Marketing, Advertising & PR']

df['cleaned_industry']= ('Other')

for i in Education_tech:
    df.loc[df['industry']== i, 'cleaned_industry'] = 'Education_tech'
for i in Gov_manufacturing:
    df.loc[df['industry']== i, 'cleaned_industry'] ='Gov_manufacturing'
for i in Media_healthcare:
    df.loc[df['industry']== i, 'cleaned_industry'] = 'Media_healthcare'
for i in Marketing:
    df.loc[df['industry']== i,'cleaned_industry'] ='Marketing'


# #### After Cleaning the Industry column we have obtained the data output in above mentioned variables which makes it easier to classify it.

# In[14]:


print('New value of Industry:',df['cleaned_industry'].unique().shape[0], '\n')

df['cleaned_industry'].value_counts()


# In[15]:


#Using the head function we can see the new column has been updated and visible. 

df.head()


# # *Age*
# As there are multiple age groups alloted in the data so we use the specific limit it.

# In[16]:


#Using Value_counts we got the age range of it. 

df['age'].value_counts()


# In[17]:


df['age_new'] = df['age']


# ### Using the Replace function to get the an individual age limit for all the age groups given

# In[18]:


df['age_new'].replace(['35-44','25-34','45-54','55-64','18-24','65 or over','under 18'],[44,34,54,64,24,65,18],inplace = True)


# In[19]:


# After conversion here is the prefered age category value count.

df['age_new'].value_counts()


# In[20]:


#Using the head function updated AGE column has been acheived 

df.head()


# # Track_Record 
# Using the previous method of Replace function we will be assigning all the ranges of record into categorised limit.

# In[21]:


#Count record 

df['track_record'].value_counts()


# In[22]:


# Grouping the experience list of each grouo to maximum number 


# In[23]:


df['overall_experience_list'] = df['track_record'] 
df['overall_experience_list'].replace(['11-20 years','21-30 years','8-10 years','5-7 years','31-40 years','2-4 years','41 years or more', '1 year or less'],[20,30,10,7,40,4,41,1],inplace=True)


# In[24]:


# After Performing the function all the values are replaced with maximum number

#which makes it simple to identify the records.


# In[25]:


df['overall_experience_list'].value_counts()


# # Field_Experience
# 
# We have to familiarise the years list of Experience by cleaning the data list with again replace function which will be effective for listing the Field experience list properly 

# In[26]:


#Finding total list of data from it.

df['field_experience'].value_counts()


# In[27]:


df['field_experience_new'] = df['field_experience'] 

df['field_experience_new'].replace(['11-20 years','21-30 years','8-10 years','5-7 years','31-40 years','2-4 years','41 years or more', '1 year or less'],[20,30,10,7,40,4,41,1],inplace=True)


# In[28]:


# Final output of updated list with maximum year selection has been performed.

df['field_experience_new'].value_counts()


# In[29]:


#New column with updated column has been displaced succesfully.

df.head()


# # Job_Function
# To clean and get the column accurate for it we have to implement the head fucntion to get the unique top 20 job fucntions. 
# 
# To make sure all the 20 functions gets equally distributed we have implemented the variables grouping function and For loop function

# In[30]:


#List of Job functions.

df['job_function'].value_counts().head(20)


# Now we have used four variables under which all the functions will be listed and to make sure that they get grouped perfectly we have implmented the for loop function with "loc" function as all the functions are datatype 
# 
# Thereby all the functions which aligns in the loop statement will automatically goes inside it and rest all the left functions gets transfered to another category named "Other"

# In[31]:


Applied_science_technology =['Education (Higher Education)','Computing or Tech','Engineering','Education (Primary/Secondary)','Science']
Operation_financial_support =['Government & Public Administration','Administration','Communications','Recruitment or HR']
Social_sector_nonprofits =['Law','Health care','Insurance','Nonprofits','Business or Consulting','Sales','Social Work']
Arts_media_culture = ['Accounting, Banking & Finance','Galleries, Libraries, Archives & Museums','Marketing, Advertising & PR','Media & Digital','Art & Design']

df['cleaned_working_division']= ('Other')

for i in Applied_science_technology:
    df.loc[df['job_function']== i, 'cleaned_working_division'] = 'Applied_science_technology'
for i in Operation_financial_support:
    df.loc[df['job_function']== i, 'cleaned_working_division'] ='Operation_financial_support'
for i in Social_sector_nonprofits:
    df.loc[df['job_function']== i, 'cleaned_working_division'] = 'Social_sector_nonprofits'
for i in Arts_media_culture:
    df.loc[df['job_function']== i,'cleaned_working_division'] ='Arts_media_culture'


# In[32]:


#Updated column is mentioned with new 

print('New value of Job Division:',df['cleaned_working_division'].unique().shape[0])


# In[33]:


#After cleaning here is the updated list of changed column values.

df['cleaned_working_division'].value_counts()


# # Gender

# In[34]:


# Unique number count of gender before any cleaning technique

df['gender'].unique().shape[0]


# In[35]:


# Name of all the unique value has been shown 

df['gender'].value_counts()


# We have to generalise the basic gender separation and by doing that it will help building the model with easier gender type.
# 
# Using the basic function of 'For' loop we have made the category of Male & Female. 

# In[36]:


Male = ['Man']
Female = ['Woman']

df['Diverse_gender_type']= ('Other')

for i in Male:
    df.loc[df['gender']== i, 'Diverse_gender_type'] = 'Male'
for i in Female:
    df.loc[df['gender']== i, 'Diverse_gender_type'] ='Female'


# In[37]:


# As its printed new value of Gender.

print('New value of Gender Section:',df['Diverse_gender_type'].unique().shape[0])


# In[38]:


# Specific updated genders have been showed below. 

df['Diverse_gender_type'].value_counts()


# In[39]:


# Updated column have been added and is visible through Head function.

df.head()


# # Race_Column
# 
# 
# To obtain the data after cleaning we will be using the "If-else Statement that contains the statement which implies as if the Attribute of race contains a comma,that means its value race is multiple one and if this is not the case then it will be considered as single uni case of Race.
# 
# 

# In[40]:


# Detailed list of all the race

df['specify_race'].value_counts()


# In[41]:


# Using the function astype it is helping to convert the input to string object.

df['specify_race'] = df['specify_race'].astype(str)


# In[42]:


def cleaned_race(race):
    
    # If the race input include a comma, it reflects multiple races
    if ',' in race:
        return 'Multiple Races'
    else:
        # Ore else sustain only the innitial race before comma
        return race.split(',')[0]

# Apply the function to the Race column
df['cleaned_race_column'] = df['specify_race'].apply(cleaned_race)


# In[43]:


# Here is every race value specific.

df['cleaned_race_column'].value_counts()


# In[44]:


# Column race have been updated and is applicable with new column in the data.

df.head()


# #  *Currency*
# 
# We are setting new variable for currency to replace with new one and the cleaning of it wil be based on looping Procedures.
# 
# In the given code we have taken For loop with iteration 'i' and two variables named curr and sal that will be used in currency and salary seperation based.
# 
# Now while cleaning the data multiple entries are in multiple currency provided so we have used If-Else condition with addition of salary rate of that exisiting salary,thereby the resultant salary will be the picture of old salary which will be eventually be converted to required USD salary.
# 
# The salary which doesnt include any numbers goes to none output in else function.

# In[45]:


df['new_currency_usd'] = None

for i in range(len(df)):
    curr = df.loc[i, 'currency']
    sal = df.loc[i, 'salary']

    if curr == 'USD':
        df.loc[i, 'new_currency_usd'] = sal
    elif curr == 'CAD':
        df.loc[i, 'new_currency_usd'] = sal * 0.73
    elif curr == 'GBP':
        df.loc[i, 'new_currency_usd'] = sal * 1.27
    elif curr == 'EUR':
        df.loc[i, 'new_currency_usd'] = sal * 1.08
    elif curr == 'AUD':
        df.loc[i, 'new_currency_usd'] = sal * 0.66
    elif curr == 'NZD':
        df.loc[i, 'new_currency_usd'] = sal * 0.60
    elif curr == 'CHF':
        df.loc[i, 'new_currency_usd'] = sal * 1.10
    elif curr == 'SEK':
        df.loc[i, 'new_currency_usd'] = sal * 0.091
    elif curr == 'JPY':
        df.loc[i, 'new_currency_usd'] = sal * 0.0066
    elif curr == 'ZAR':
        df.loc[i, 'new_currency_usd'] = sal * 0.055
    elif curr == 'HKD':
        df.loc[i, 'new_currency_usd'] = sal * 0.13
    else:
        df.loc[i, 'new_currency_usd'] = None


# In[46]:


# Currency Counts after changes

df['new_currency_usd'].value_counts()


# # Foreign_Currency
# Each value of the currency which have multiple input names will be considered as Foreign currency and there by will be cleaned thoroughly.

# In[47]:


# Printing all the Other currency which contain unique value name.
df['foreign_currency'].value_counts()


# Now that we have so many text formatted data in form of currency we will be using the functions named 'str' and "replace" so that all the text formatted value transforms into a particular currency which will be easier to use in model prediction.

# In[48]:


other = df['currency'] == 'Other'

# Replacing the text with NAN values/ Non currency values
df.loc[other & df['foreign_currency'].isin(['NA', 'THE $6K IS FOR BUYING HEALTH INSURANCE ON THE MARKETPLACE AND IS TAXED AS NORMAL INCOME. ONLY THE $48600 IS USED FOR CALCULATING RAISES.', 'I ALSO RECEIVE $20,000-30,000 IN STOCK COMPENSATION', 'I WILL RECEIVE A BONUS BUT UNSURE OF THE AVERAGE', 'BONUS VARIES BASED ON INDIVIDUAL AND COMPANY PERFORMANCE','80000', 'BONES']), 'new_currency_usd'] = None

df['foreign_currency'] = df['foreign_currency'].str.upper()


#Updating the values with new input provided.
df['foreign_currency'] = df['foreign_currency'].replace({
    
    'NZD (NEW ZEALAND DOLLARS)': 'NZD',
    'ILS (ISRAELI NEW SHEKEL)': 'ILS',
    'DKK (DANISH KRONER)': 'DKK',
    'ISK  ICELANDIC KRONA': 'ISK',
    'TRINIDAD AND TOBAGO DOLLAR': 'TTD',
    'SGD (SINGAPORE DOLLARS)': 'SGD',
    'SINGAPORE DOLLAR': 'SGD',
    'SGD ': 'SGD',
    'SGD (SINGAPORE DOLLARS)': 'SGD',   
})


# ## Foreign_Currency conversion
# 
# As we have converted all the values with real data.Now to make a list from that data in USD dollar we will be using the Loop Statement that will be helpfull for converting the multiple currency with real Exchange rate and we will be getting the new format data that would be already converted and easy to use.

# In[49]:


# 'i' will be reading all the values in that mentioned column.
for i in range(len(df)):
    other_currency = df.loc[i, 'foreign_currency']
    sal = df.loc[i, 'salary']

# Using the If-Else will be changing the value and converting currency conversion.

    if other_currency == 'DKK':
        df.loc[i, 'new_currency_usd'] = sal * 0.1446
    elif other_currency == 'NOK':
        df.loc[i, 'new_currency_usd'] = sal * 0.0937
    elif other_currency == 'SGD':
        df.loc[i, 'new_currency_usd'] = sal * 0.746
    elif other_currency == 'KRW':
        df.loc[i, 'new_currency_usd'] = sal * 0.00069
    elif other_currency == 'INR':
        df.loc[i, 'new_currency_usd'] = sal * 0.012
    elif other_currency == 'CHF':
        df.loc[i, 'new_currency_usd'] = sal * 1.12
    elif other_currency == 'NIS':
        df.loc[i, 'new_currency_usd'] = sal * 0.27
    elif other_currency == 'NZD':
        df.loc[i, 'new_currency_usd'] = sal * 0.60
    elif other_currency == 'HUF':
        df.loc[i, 'new_currency_usd'] = sal * 0.0027
    elif other_currency == 'BRL':
        df.loc[i, 'new_currency_usd'] = sal * 0.19
    elif other_currency == 'PLN':
        df.loc[i, 'new_currency_usd'] = sal * 0.25
    elif other_currency == 'ZMW':
        df.loc[i, 'new_currency_usd'] = sal * 0.055
    elif other_currency == 'ILS':
        df.loc[i, 'new_currency_usd'] = sal * 0.27
    elif other_currency == 'ISK':
        df.loc[i, 'new_currency_usd'] = sal * 0.0068
    elif other_currency == 'SEK':
        df.loc[i, 'new_currency_usd'] = sal * 0.094
    elif other_currency == 'TTD':
        df.loc[i, 'new_currency_usd'] = sal * 0.15
    else:
        df.loc[i, 'new_currency_usd'] = None


# In the table of foreign currency there were mulitple NAN values which we have overcome with 'Dropna',
# By using this all the values which contains the NAN values have been removed and data has been cleared from before 

# In[50]:


df['foreign_currency'].value_counts()


# In[51]:


# Using the greater then zero condition for Negative or zero value
df = df[df['new_currency_usd'] > 0]

#Using this for eliminating NAN values from new column.
df = df.dropna(subset=['new_currency_usd'])


# In[52]:


#Updated row has been displayed.
df.head()


# # Workplace
# 
# We will be segragating the working countries into their respective Grouped assigned to them with using the For loop statement that contains 

# In[53]:


df['workplace'].unique()


# In[54]:


#Variables which will store all countries under it.

United_States_america = ['United States']
Canada = ['Canada']
Unitedkingdom_Ireland = ['United Kingdom','Ireland']
Oceania = ['Australia','New Zealand']
Western_Europe = ['Germany','Netherlands','France','Switzerland','Belgium','Spain','Denmark','Sweden']
Japan = ['Japan']


df['Other_country_type']= ('Other')

for i in United_States_america:
    df.loc[df['workplace']== i, 'Other_country_type'] = 'United_States_america'
for i in Canada:
    df.loc[df['workplace']== i, 'Other_country_type'] ='Canada'
for i in Unitedkingdom_Ireland:
    df.loc[df['workplace']== i, 'Other_country_type'] ='Unitedkingdom_Ireland'
for i in Oceania:
    df.loc[df['workplace']== i, 'Other_country_type'] ='Oceania'
for i in Western_Europe:
    df.loc[df['workplace']== i, 'Other_country_type'] ='Western_Europe'
for i in Japan:
    df.loc[df['workplace']== i, 'Other_country_type'] ='Japan'


# In[55]:


# Unique Values of the Countries have been 
print('New unique place list:',df['workplace'].unique().shape[0])


# In[56]:


# Updated Column output have been shown.
df['Other_country_type'].value_counts()


# # Education
# 
# By targeting the top 15 qualification with grouping function data cleaning have been performed.
# 
# Organising the data into three new variables and storing it into each of it and those which are out of that will be direclty termed as 'others'

# In[57]:


# Printing the first 15 value of qualification

df['qualification'].value_counts().head(15)


# In[58]:


# Assigning the variables

Postgraduate_Degree = ["Master's degree",'Masters + CPA licensure','Doctorate','Professional degree (MD, JD, etc.)']
Senior_Secondary = ['College degree','Some college',"Associate's degree","Bachelor's degree",'Graduate Diploma','BSc',"Bachelor's Degree",'Graduate certificate']
Secondary_school = ['High School','Trade School','CPA']

df['Other_degree_type']= ('Other')

# Checking and adding the value into Variables respectively.

for i in Postgraduate_Degree:
    df.loc[df['qualification']== i, 'Other_degree_type'] = 'Postgraduate_Degree'
for i in Senior_Secondary:
    df.loc[df['qualification']== i, 'Other_degree_type'] ='Senior_Secondary'
for i in Secondary_school:
    df.loc[df['qualification']== i, 'Other_degree_type'] ='Secondary_school'


# In[59]:


#Updated list of qualification is printed

print('New value of Degree:',df['Other_degree_type'].unique().shape[0])


# In[60]:


# New list with improved name have been listed 

df['Other_degree_type'].value_counts()


# # Code for assigning all the salary under Salary Range
# 
# "cut" Function allows to fill all the values of the column into the unique range group provided.
# 

# In[61]:


# Displaying the salary into three ranges.
df['grouped_salary'] = pd.cut(
    df['new_currency_usd'],
    bins=[0, 70000, 100000, float('inf')],
    labels=['Low', 'Medium', 'High']
)

#Authorising the data type of the column 
df['grouped_salary'].dtype

#Converting the colummn into User friendly categorical data with 'astype func.'
df['grouped_salary'] = df['grouped_salary'].astype('category')

# Printing the Cleaned exceptionally value
print(df['grouped_salary'].unique())

# Prints all the Nan values and adding up together.
print(df['grouped_salary'].isna().sum())


# ### K-means Algorithm Utilisation with Agglomerative Clustering

# In[62]:


#Bringing LabelEncoder class via sklearn.preprocessing Module.
# from sklearn.preprocessing import LabelEncoder


# In[63]:


#Importing AgglomerativeClustering through sklearn.cluster Module.
# from sklearn.cluster import AgglomerativeClustering


# In[64]:


# Erasing all the Empty values.
# df = df.dropna(subset=['hybrid_on_site'])


# In[65]:


# Using syntax 'le' for LabelEncoder Class
# le = LabelEncoder()

# #Converting the values of Mode_encoded column into numeric-0:1:2
# # Fitting the module in it.
# df['mode_encoded'] = le.fit_transform(df['hybrid_on_site'])


# In[66]:


# Apply Hierarchical Clustering
# hc = AgglomerativeClustering(n_clusters=4)


# In[67]:


# df['Cluster_hirearchial'] = hc.fit_predict(df[['mode_encoded']])

# # Show result
# print(df[['hybrid_on_site', 'mode_encoded', 'Cluster_hirearchial']].head())


# In[68]:


# Drop blank values
# df = df.dropna(subset=['organised'])


# In[69]:


# Encode text labels
# le = LabelEncoder()
# df['mode_encoded'] = le.fit_transform(df['organised'])


# In[70]:


# Apply Hierarchical Clustering
# hc = AgglomerativeClustering(n'_clusters=4)


# In[71]:


# df['Cluster_hirearchial'] = hc.fit_predict(df[['mode_encoded']])

# # Show result
# print(df[['organised', 'mode_encoded', 'Cluster_hirearchial']].head(5))


# # Algorithm Selection

# In[72]:


# General details of all columns listed.
df.info()


# ### Importing all the mentioned required Functions and Sets.

# In[73]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


# In[74]:


# Columns to use for prediction
columns_to_use = [
    
    ## "Target"
    "grouped_salary",           

    ## "Predictors"
    "age_new",
    "cleaned_industry",
    "cleaned_working_division",
    "hybrid_on_site",
    "organised",
    "overall_experience_list",
    "field_experience_new",
    "Diverse_gender_type",
    "cleaned_race_column",
    "Other_degree_type",
]


# In[75]:


# "Describe Target & Predictors"
df_predi = df[columns_to_use].copy()


y = df_predi["grouped_salary"]              # target
X = df_predi.drop("grouped_salary", axis=1) # predictors


# In[76]:


# 'Encrypt Categorical Columns'

label_cols = [
    "cleaned_industry",
    "cleaned_working_division",
    "Other_degree_type",
    "Diverse_gender_type",
    "cleaned_race_column",
    "hybrid_on_site",
    "organised"
]

## "Apply Label Encoding to categorical columns with Described categories"
label_cols = [ col for col in label_cols if col in X.columns]
le = LabelEncoder()
for col in label_cols:
    X[col] = le.fit_transform(X[col].astype(str))


# In[77]:


## "Train-Test Validation Split "
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=0
)


# In[78]:


## "Specifying Models"
model1 = DecisionTreeClassifier(random_state = 0)
model2 = KNeighborsClassifier()
model3 = LogisticRegression(max_iter=10000, solver='lbfgs')


# In[79]:


##'Using Voting Ensemble Classifier'
clf_ensemble = VotingClassifier(
    estimators=[('dt', model1), ('knn', model2), ('lr', model3)],
    voting='hard'
)


# In[80]:


##"Fit the Ensemble Model"
clf_ensemble.fit(X_train, y_train)


# In[81]:


#'Predicting the Model'
y_train_pred = clf_ensemble.predict(X_train)
y_test_pred = clf_ensemble.predict(X_test)


# In[82]:


# "Printing Accuracy Score "
train_score = accuracy_score(y_train, y_train_pred)
test_score = accuracy_score(y_test, y_test_pred)

print("Train/Test Accuracy: %.3f / %.3f" % (train_score, test_score))


# # Solving Required Questions Using Following CLassifier.
# 
# 1. Which factors most strongly influence annual income globally?
# 
# ### For solving this question we will be implementing Random forest Regressor model and the model wil predict and demonstrate the visualization of the target column and predictor column. 

# In[83]:


# Prefer only choicable target column
Regress_col_out = [
    "new_currency_usd",              
    "hybrid_on_site", 
    "organised",
    "age_new",
    "cleaned_industry",
    "cleaned_working_division",
    "Other_country_type",
    "overall_experience_list",
    "field_experience_new",
    "Other_degree_type",
    "Diverse_gender_type",
    "cleaned_race_column"
]
 
df_reg = df[Regress_col_out].copy()


# In[84]:


# 'Predictors'
y_reg = df_reg["new_currency_usd"]
X_reg = df_reg.drop("new_currency_usd", axis=1)


# In[85]:


# 'Categorical Feature'
label_cols = [
    "cleaned_industry",
    "cleaned_working_division",
    "Other_country_type",
    "Other_degree_type",
    "Diverse_gender_type",
    "cleaned_race_column",
    "hybrid_on_site",
    "organised"
]
 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
 
for col in label_cols:
    X_reg[col] = le.fit_transform(X_reg[col].astype(str))


# In[86]:


# 'Train_Test_Split Model'
from sklearn.model_selection import train_test_split
 
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=0
)


# In[87]:


# "Utilisation of RandomForestRegressor"
from sklearn.ensemble import RandomForestRegressor
 
rf = RandomForestRegressor(
    n_estimators=300,
    random_state=0
)
 
rf.fit(X_train_r, y_train_r)


# In[88]:


# "Plotting through library Seaborn"
import seaborn as sns
import pandas as pd
importances = rf.feature_importances_
feature_names = X_reg.columns

df_imp = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=df_imp, x='Importance', y='Feature')
plt.title("\nHighlighted Feature\n")
plt.show()


# # 2nd Question-: Can we predict an individual’s income range based on their attribute
# 
# To solve it we will be using the RandomForestClassifier module that will build and predict the accuracy with column required and also help us finds the accuracy model will predict irrespctive of all the features asked.

# In[89]:


# Importing RandomForestClassifier and LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[90]:


# 'Income range Predictors'
Guess_cols = [
    "grouped_salary",
    "hybrid_on_site", 
    "organised",
    "age_new",
    "cleaned_industry",
    "cleaned_working_division",
    "overall_experience_list",
    "field_experience_new",
    "Other_degree_type",
    "Diverse_gender_type",
    "cleaned_race_column"
]
 
df_income = df[Guess_cols].copy()


# In[91]:


# "Assigned name for Target values"
y = df_income["grouped_salary"]
x = df_income.drop("grouped_salary", axis=1)


# In[92]:


# "Using LabelEncoder Feature module"
categorical_cols = [
    "cleaned_industry",
    "cleaned_working_division",
    "Other_degree_type",
    "Diverse_gender_type",
    "cleaned_race_column",
    "hybrid_on_site",
    "organised"
]
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le


X_train_inc, X_test_inc, y_train_inc, y_test_inc = train_test_split(
    X, y, test_size=0.3, random_state=0)

# Printing accurate column in test dataset
X_test_inc = X_test_inc[X_train_inc.columns]

print ("Train X:", X_train_inc.shape)
print ("Train y:", y_train_inc.shape)
print ("Test X:", X_train_inc.shape)
print ("Test y:", y_train_inc.shape)


# In[93]:


# Using the RandomForestClassifier 
income_op = RandomForestClassifier(n_estimators=300, random_state=0)
income_op.fit(X_train_inc, y_train_inc)


# In[94]:


y_train_pred = income_op.predict(X_train_inc)
y_test_pred = income_op.predict(X_test_inc)


# In[95]:


train_accuracy = accuracy_score(y_train_inc, y_train_pred)
test_accuracy = accuracy_score(y_test_inc, y_test_pred)

print("Train accuracy is",train_accuracy)
print("Test accuracy is",test_accuracy)


# In[96]:


#"Final Code output for salary range prediction"
Predictor_value = pd.DataFrame(columns = X.columns)

Predictor_value.loc[0] = [
    1,
    0,
    30,
    X["cleaned_industry"].iloc[0],
    X["cleaned_working_division"].iloc[0],
    5,
    3,
    X["Other_degree_type"].iloc[0],
    X["Diverse_gender_type"].iloc[0],
    X["cleaned_race_column"].iloc[0]
]
 
Range_value = income_op.predict(Predictor_value)
print("Predicted Income Range:", Range_value[0])


# In[97]:


from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Confusion Matrix
con_mat = confusion_matrix(y_test_inc, y_test_pred)

plt.figure(figsize=(6,4))
sns.heatmap(con_mat, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix for Salary Grouping Prediction")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test_inc, y_test_pred))


# So to display the visualisation of the salary range that has been predicted we will be writing the small code that will be based on stacked bar chart that will plot two plot that demonstrates the salary with respect to gender.We have used function median and mean that will find all the individual data's salary with column gender 

# In[98]:


# Importing library numpy 
import numpy as np

# "Allot the data by gender"
group_gender = df.groupby("Diverse_gender_type")

median_salary_gender = group_gender["new_currency_usd"].median()
mean_salary_gender = group_gender["new_currency_usd"].mean()

# 'Printing the output seperately'
print(median_salary_gender,'\n')
print(mean_salary_gender)

## 'Gender Naming'
labels = median_salary_gender.index

# 'Convert value function'
median_values = median_salary_gender.values
mean_values = mean_salary_gender.values

## 'Plotting on x-axis'


x = np.arange(len(labels))

plt.figure(figsize=(10,6))

## "Plotting Stacked"
plt.bar(x, median_values, label="Median Salary", edgecolor="black")
plt.bar(x, mean_values, bottom=median_values, label="Mean Salary", edgecolor="black")

plt.xticks(x, labels)
plt.ylabel("Salary USD")
plt.title("Median + Mean Salary by Gender Stacked")

plt.legend()
plt.tight_layout()
plt.show()


# # Qn 3 Are there visible gender or regional pay disparities in the data
# In order to understand the disparity we have taken the overall experience list column that will determine us the plotting of data with respect to salary column.
# 
# We have implemented the plotting using library seaborn and matplotlib.pyplot
# 

# In[99]:


## "Importing required libraries"
import seaborn as sns
import matplotlib.pyplot as plt


# In[100]:


# Group and calculate median salary
groupby_country = df.groupby("overall_experience_list")
median_salary_country = groupby_country["new_currency_usd"].median()

## "Convert to DataFrame and Arrange from highest to lowest value"
median_salary_country_sorted = (
    median_salary_country
    .sort_values(ascending=False)
    .reset_index()
)


# In[101]:


## "Print the values as per data"
print(median_salary_country_sorted, '\n')

## "Implementing barplot by seaborn"
plt.figure(figsize=(14, 6))
sns.barplot(
    data=median_salary_country_sorted,
    x="overall_experience_list",
    y="new_currency_usd",
    palette="viridis"
)


# In[ ]:





# In[ ]:




