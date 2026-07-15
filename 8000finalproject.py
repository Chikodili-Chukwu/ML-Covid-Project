#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install scikit-learn')

get_ipython().system('pip install -U scikit-learn')

get_ipython().system('pip3 install scikit-learn')


# In[56]:


import pandas as pd 
import numpy as np
import seaborn as sns

#import necessary things


# In[57]:


data = pd.read_csv(r"C:\Users\chibz\OneDrive\Desktop\CSC 8000 (Python)\compact.csv")

#to import the file


# In[4]:


africa_data = data[data['continent'].str.lower() == 'africa']

#streamlined covid data to only african countries


# In[5]:


africa_data.info()

#information about the dataset


# In[6]:


africa_data = africa_data.dropna(axis=1, how="all")

#drop columns that were all empty


# In[7]:


africa_data.info()


# In[8]:


africa_data.shape

#to know the total rows and columns


# In[9]:


africa = africa_data.drop(columns=["code", "excess_mortality_cumulative",
                                   "excess_mortality_cumulative_absolute", 
                                  "excess_mortality_cumulative_per_million",
                                  "hosp_patients","hosp_patients_per_million" ,"weekly_hosp_admissions", 
                                  "weekly_hosp_admissions_per_million" , "icu_patients" , 
                                  "icu_patients_per_million" ,"excess_mortality" ])

#dropped columns less than 50% of the total dataset


# In[10]:


africa.info()


# In[11]:


africa = africa.drop(columns=["total_vaccinations", "people_vaccinated",
                                   "people_fully_vaccinated", 
                                  "total_boosters",
                                  "new_vaccinations","total_vaccinations_per_hundred" ,"people_vaccinated_per_hundred", 
                                  "total_boosters_per_hundred" ])


#dropped columns with little data


# In[12]:


africa.info()


# In[13]:


africa[["new_tests_smoothed_per_thousand", "positive_rate"]].corr()

#checking for corrrelation to know what to drop.


# In[14]:


africa[["new_tests_smoothed_per_thousand", "new_cases_smoothed_per_million"]].corr()


# In[15]:


africa[["positive_rate", "new_cases_smoothed_per_million"]].corr()


# In[16]:


sns.regplot(
    data=africa,
    x="new_tests_smoothed_per_thousand",
    y="new_cases_smoothed_per_million"
)


# In[17]:


africa = africa.drop(columns=["total_tests", "new_tests",
                                   "total_tests_per_thousand", 
                                  "new_tests_per_thousand",
                                  "new_tests_smoothed","new_tests_smoothed_per_thousand" ,"positive_rate", 
                                  "tests_per_case" , "people_fully_vaccinated_per_hundred" , 
                                   ])



# In[18]:


africa.info()


# In[19]:


africa.head()

#to see the first 5 rows of the dataset


# In[20]:


africa["date"] = pd.to_datetime(africa["date"])


#changing the date to timeseries 


# In[21]:


africa['year'] = africa['date'].dt.year


#adding new column of all year


# In[22]:


cols_to_fill = ['total_cases', 'new_cases', 'new_deaths' , 'new_cases_smoothed' , 
               'total_cases_per_million' , 'new_cases_per_million',
               'new_cases_smoothed_per_million' , 'total_deaths', 
               'new_deaths_smoothed' , 'total_deaths_per_million' ,
               'new_deaths_per_million' ,'new_deaths_smoothed_per_million',
               'gdp_per_capita' ,'extreme_poverty', 'diabetes_prevalence', 
               'handwashing_facilities']

africa[cols_to_fill] = (
    africa
    .groupby(['country', 'year'])[cols_to_fill]
    .transform(lambda x: x.fillna(x.median()))
)


#filling null values with group by function of both country and the year


# In[23]:


africa.info()


# In[24]:


cols = ['total_cases', 'new_cases', 'new_deaths' , 'new_cases_smoothed' , 
               'total_cases_per_million' , 'new_cases_per_million',
               'new_cases_smoothed_per_million' , 'total_deaths', 
               'new_deaths_smoothed' , 'total_deaths_per_million' ,
               'new_deaths_per_million' ,'new_deaths_smoothed_per_million',
               'gdp_per_capita' ,'extreme_poverty', 'diabetes_prevalence', 
               'handwashing_facilities']   

africa[cols] = (
    africa
    .groupby('country')[cols]
    .transform(lambda x: x.fillna(x.median()))
)


africa[cols] = africa[cols].fillna(africa[cols].median())



#filled with only country and then with all the continent


# In[25]:


africa.info()


# In[26]:


cols1= ['stringency_index', 'reproduction_rate', 'new_vaccinations_smoothed' , 
         'new_vaccinations_smoothed_per_million' , 
               'new_people_vaccinated_smoothed' , 
                 'new_people_vaccinated_smoothed_per_hundred',
               'hospital_beds_per_thousand']   

africa[cols1] = (
    africa
    .groupby(['country', 'year'])[cols1]
    .transform(lambda x: x.fillna(x.median()))
)

africa[cols1] = (
    africa
    .groupby('country')[cols1]
    .transform(lambda x: x.fillna(x.median()))
)


africa[cols1] = africa[cols1].fillna(africa[cols1].median())


#filled with only country & year and then with all the continent


# In[27]:


africa.info()


# In[28]:


africa.describe()


# In[29]:


africa['country'].unique()


# I want to group them based on region. 


# In[51]:


region_map = {
    
    
    # Northern Africa
    'Algeria': 'Northern Africa',
    'Egypt': 'Northern Africa',
    'Libya': 'Northern Africa',
    'Morocco': 'Northern Africa',
    'Sudan': 'Northern Africa',
    'Tunisia': 'Northern Africa',
    'Western Sahara': 'Northern Africa',
    'Mayotte': 'Northern Africa',

    # Western Africa
    'Benin': 'Western Africa',
    'Burkina Faso': 'Western Africa',
    'Cape Verde': 'Western Africa',
    "Cote d'Ivoire": 'Western Africa',
    'Gambia': 'Western Africa',
    'Ghana': 'Western Africa',
    'Guinea': 'Western Africa',
    'Guinea-Bissau': 'Western Africa',
    'Liberia': 'Western Africa',
    'Mali': 'Western Africa',
    'Mauritania': 'Western Africa',
    'Niger': 'Western Africa',
    'Nigeria': 'Western Africa',
    'Senegal': 'Western Africa',
    'Sierra Leone': 'Western Africa',
    'Togo': 'Western Africa',
    'Saint Helena': 'Northern Africa', 
    
    # Eastern Africa
    'Burundi': 'Eastern Africa',
    'Comoros': 'Eastern Africa',
    'Djibouti': 'Eastern Africa',
    'Eritrea': 'Eastern Africa',
    'Ethiopia': 'Eastern Africa',
    'Kenya': 'Eastern Africa',
    'Madagascar': 'Eastern Africa',
    'Malawi': 'Eastern Africa',
    'Mauritius': 'Eastern Africa',
    'Mozambique': 'Eastern Africa',
    'Reunion': 'Eastern Africa',
    'Rwanda': 'Eastern Africa',
    'Seychelles': 'Eastern Africa',
    'Somalia': 'Eastern Africa',
    'South Sudan': 'Eastern Africa',
    'Tanzania': 'Eastern Africa',
    'Uganda': 'Eastern Africa',

    # Central Africa
    'Angola': 'Central Africa',
    'Cameroon': 'Central Africa',
    'Central African Republic': 'Central Africa',
    'Chad': 'Central Africa',
    'Congo': 'Central Africa',
    'Democratic Republic of Congo': 'Central Africa',
    'Equatorial Guinea': 'Central Africa',
    'Gabon': 'Central Africa',
    'Sao Tome and Principe': 'Central Africa',

    # Southern Africa
    'Botswana': 'Southern Africa',
    'Eswatini': 'Southern Africa',
    'Lesotho': 'Southern Africa',
    'Namibia': 'Southern Africa',
    'South Africa': 'Southern Africa',
    'Zambia': 'Southern Africa',
    'Zimbabwe': 'Southern Africa'
}


africa['Region'] = africa['country'].map(region_map)


#categorizing into regions


# In[31]:


africa.info()


# In[32]:


import matplotlib as mpl
import matplotlib.pyplot as plt


# In[33]:


plt.figure(figsize=(9,6))
sns.barplot(data=africa, x="Region", y="total_cases_per_million", estimator='mean',  
            hue='Region')
plt.title("Average Total Cases Per Million by Region")
plt.show()


#finding information based on regions


# In[34]:


top10_cases = (
    africa.groupby("country")["total_cases_per_million"]
          .max()                      # each country has multiple rows so take the max
          .sort_values(ascending=False)
          .head(5)
)

top10_deaths = (
    africa.groupby("country")["total_deaths_per_million"]
          .max()
          .sort_values(ascending=False)
          .head(5)
)

plt.subplot(2, 1, 1)
plt.bar(top10_cases.index, top10_cases.values, color="blue")
plt.title("Top 5 African Countries With Highest COVID-19 Cases")
plt.ylabel("Total Cases")
plt.figure(figsize=(16, 10));

print (   )

plt.subplot(2, 1, 2)
plt.bar(top10_deaths.index, top10_deaths.values, color="red")
plt.title("Top 5 African Countries With Highest COVID-19 Deaths")
plt.ylabel("Total Deaths")
plt.figure(figsize=(16, 10));





# In[35]:


top5_countries = (
    africa.groupby("country")["total_cases"]
          .max()
          .sort_values(ascending=False)
          .head(5)
          .index
)
top5_countries


plt.figure(figsize=(12,6))

for country in top5_countries:
    subset = africa[africa["country"] == country]
    plt.plot(subset["date"], subset["total_cases"], label=country)

plt.title("COVID-19 Total Cases Over Time (Top 5 African Countries)")
plt.xlabel("Date")
plt.ylabel("Total Cases")
plt.legend()
plt.show()


#information about covid's cases trend


# In[37]:


africa = africa[africa["date"].dt.year <= 2022]


#I dropped information later than 2022 cause I noticed from 2022, there was no new growth in cases


# In[38]:


africa.info()


# In[50]:


numeric_df = africa.select_dtypes(include=['float64', 'int64'])
#utilizing only numerical data

plt.figure(figsize=(14,12))
sns.heatmap(numeric_df.corr(), cmap='viridis', annot=True, fmt=".2f" )
            #"fmt this is for decimal place 
plt.title("Correlation Heatmap of Numeric COVID Features")
plt.show()


# In[53]:


print(africa.columns)


# In[59]:


africa['log_cases'] = np.log1p(africa['total_cases_per_million'])
africa['log_deaths'] = np.log1p(africa['total_deaths_per_million'])
africa['log_gdp'] = np.log1p(africa['gdp_per_capita'])


# In[61]:


country_df = africa.groupby('country').agg({
    'log_gdp':'mean',
    'log_cases':'max',
    'log_deaths':'max',
    'diabetes_prevalence':'mean',
    'Region':'first'
}).reset_index()


# In[64]:


sns.pairplot(
    country_df[['log_gdp', 'log_cases', 'log_deaths', 'diabetes_prevalence', 'Region']],
    hue='Region'
)
plt.show()


# In[65]:


numeric_cols = africa.select_dtypes(include=['float64','int64']).columns

skewed = africa[numeric_cols].skew().sort_values(ascending=False)
skewed


# In[67]:


skewed_cols = skewed[skewed > 1].index.tolist()
skewed_cols


for col in skewed_cols:
    africa['log_' + col] = np.log1p(africa[col])


# In[69]:


africa.info()


# In[ ]:




