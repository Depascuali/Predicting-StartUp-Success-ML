# Predicting-StartUp-Success-ML
Final project for university consisting of creating a machine learning model to determine the success of a Start-Up. In this case, success is defined as the Start-Up reaching an investment round beyond the Seed stage.

# Introduction
For this project, we have various tables containing data related to Start-Ups around the world. Here you can see an overview of the data diagram:

![image](https://github.com/Depascuali/Predicting-StartUp-Success-ML/assets/97790973/98ed95df-72a7-4f79-a443-6409cc1c63d6)


The process will be as follows:

1) We will select the case study.
2) Conduct a data exploration to understand data types, inconsistencies, null values, etc.
3) After the exploration, we will go through each table performing data cleaning and transformation.
4) Once the transformation is done, we will conduct further analysis to gain insights from the resulting tables.
5) Finally, we will compile a table with the dimensions to be used in the ML model.
6) We will test different models to choose the one with the highest accuracy.
7) We will provide a conclusion about the results.

# Case Study
We have decided to focus on the "Investment Stage" variable, which refers to the stage reached by the start-up during a specific investment round. There are four possible values in this column: "Seed," "Early Stage Venture," "Late Stage Venture," and "Private Equity." Our goal will be to analyze the probability that a Start-Up progresses beyond the Seed stage given the data provided in the tables. We believe this model could be useful to risk-averse investment funds that are looking for more established companies that have already passed the initial investment stage. This approach would help these funds to identify potential investment opportunities more effectively, by focusing on start-ups that have demonstrated some degree of success and stability beyond the volatile Seed stage.

# Data Exploration
*(The file is also available in the repository: Data Exploration - Group G.ipynb)*

We install necesary libraries
```python
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
import matplotlib.dates as mdates
from tabulate import tabulate
!pip install pandas-profiling==3.0.0
```

We assign each data set to a table
```python
# Excel file path
file_path = "/content/drive/MyDrive/applied data science research project/2324RX19_Project_Data.xlsx"

# Load the Excel file
xls = pd.ExcelFile(file_path)
```
```python
# Create separate dataframes for each sheet
# Load the Excel file, specifying that dashes "-" should be interpreted as NaN
company_data = pd.read_excel(file_path, sheet_name='01.COMPANY', na_values='—')
investment_data = pd.read_excel(file_path, sheet_name='02.INVESTMENT', na_values='—')
acquisition_data = pd.read_excel(xls, '03.ACQUISITION', na_values='—')
employee_data = pd.read_excel(xls, '04.EMPLOYEE')
news_data = pd.read_excel(xls, '05.NEWS')
```

### *Company Table*
```python
# Show basic information of company_data
print("\nSheet: 01.COMPANY")

# Transpose and describe the company_data
transposed_company_data_description = company_data.describe().transpose()

# Add the data types to the transposed table
transposed_company_data_description['Data Type'] = company_data.dtypes

# Convert the transposed description to a structured table
table = tabulate(transposed_company_data_description, headers='keys', tablefmt='pretty')

# Number of variables
num_variables = len(company_data.columns)

# Number of records
num_records = len(company_data)

# Null values
null_counts = company_data.isnull().sum()
null_values_table = tabulate(pd.DataFrame(null_counts, columns=['Null Count']), headers=['Variable', 'Null Count'], tablefmt='grid')

# Print the tables
print(table)

print("Number of variables:", num_variables)

print("Number of records:", num_records)

print("Null values:")
print(null_values_table)
```
```python
# Display detailed information of the DataFrame
print("\nDetailed information of the DataFrame:")
print(company_data.info())
```

### *Investment Table*
```python
# Show basic information of investment_data
print("\nSheet: 02.INVESTMENT")

# Transpose and describe the investment_data
transposed_investment_data_description = investment_data.describe().transpose()

# Add the data types to the transposed table
transposed_investment_data_description['Data Type'] = investment_data.dtypes

# Convert the transposed description to a structured table
table = tabulate(transposed_investment_data_description, headers='keys', tablefmt='pretty')

# Print the structured table



# Number of variables
num_variables = len(investment_data.columns)

# Number of records
num_records = len(investment_data)

# Null values
null_counts = investment_data.isnull().sum()
null_values_table = tabulate(pd.DataFrame(null_counts, columns=['Null Count']), headers=['Variable', 'Null Count'], tablefmt='grid')

# Statistical description
statistical_description_table = tabulate(investment_data.describe().transpose(), headers='keys', tablefmt='grid')

# Print the tables
print(table)
print("Number of variables:", num_variables)
print("Number of records:", num_records)
print("Null values:")
print(null_values_table)
```
```python
# Frequency of each FUNDING_TYPE
funding_type_counts = investment_data['FUNDING_TYPE'].value_counts()

print("Frequency of each FUNDING_TYPE:")
print(funding_type_counts)
```
```python
# Frequency of each FUNDING_TYPE plot
funding_type_counts = investment_data['FUNDING_TYPE'].value_counts()

# Create the bar chart
plt.figure(figsize=(12, 8))
funding_type_counts.plot(kind='bar', color='skyblue')

# Set the title and axis labels
plt.title('Frequency of each FUNDING_TYPE')
plt.xlabel('FUNDING_TYPE')
plt.ylabel('Frequency')

# Rotate the x-axis labels for better visibility
plt.xticks(rotation=90)

# Show the plot
plt.show()
```
```python
# Count the frequency of each INVESTMENT_STAGE, including NaN
investment_stage_counts = investment_data['INVESTMENT_STAGE'].value_counts(dropna=False)

print("Frequency of each INVESTMENT_STAGE (including NaN):")
print(investment_stage_counts)
```
```python
# Create the bar plot
plt.figure(figsize=(10, 6))
investment_stage_counts.plot(kind='bar', color='skyblue')

# Set the title and labels
plt.title('Frequency of each INVESTMENT_STAGE (including NaN)')
plt.xlabel('INVESTMENT_STAGE')
plt.ylabel('Frequency')

# Show the plot
plt.show()
```
```python
# Convert 'ANNOUNCED_DATE' column to datetime
investment_data['ANNOUNCED_DATE'] = pd.to_datetime(investment_data['ANNOUNCED_DATE'])

# Extract year from 'ANNOUNCED_DATE'
investment_data['Year'] = investment_data['ANNOUNCED_DATE'].dt.year

# Group data by 'Year' and 'INVESTMENT_STAGE'
grouped_data = investment_data.groupby(['Year', 'INVESTMENT_STAGE']).size().unstack(fill_value=0)

# Plot each group separately
plt.figure(figsize=(12, 8))
for column in grouped_data.columns:
    plt.plot(grouped_data.index, grouped_data[column], label=column)

# Set labels and title
plt.xlabel('Year')
plt.ylabel('Quantity')
plt.title('Quantity of Investment Stages Over Years')
plt.legend()

# Show plot
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### *Acquisition Table*
```python
# Show basic information of the ACQUISITION sheet
print("\nSheet: 03.ACQUISITION")

# Describe and transpose acquisition_data
acquisition_description = acquisition_data.describe().transpose()

# Add data types to the description
acquisition_description['Data Type'] = acquisition_data.dtypes

# Convert the transposed description to a table
table = tabulate(acquisition_description, headers='keys', tablefmt='pretty')

# Print the table
print(table)

# Number of variables
print("Number of Variables:", len(acquisition_data.columns))

# Number of records
print("Number of Records:", len(acquisition_data))

# Null values
null_counts = acquisition_data.isnull().sum()

# Create a summary table for null values
summary_table = []
for column, null_count in null_counts.items():
    summary_table.append([column, null_count])

# Convert the summary table to a table
table2 = tabulate(summary_table, headers=['Variable', 'Null Count'], tablefmt='pretty')

# Print the table
print(table2)
```
```python
# Count the occurrences of each ACQUISITION_TYPE, including null values
acquisition_type_counts = acquisition_data['ACQUISITION_TYPE'].value_counts(dropna=False)

print("Count of each ACQUISITION_TYPE (including null values):")
print(acquisition_type_counts)
```
```python
# Create a bar chart for the quantities of each ACQUISITION_TYPE
plt.figure(figsize=(10, 6))
acquisition_type_counts.plot(kind='bar')
plt.title('Quantity of Each Acquisition Type')
plt.xlabel('Acquisition Type')
plt.ylabel('Quantity')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```
```python
# Convert the 'ANNOUNCED_DATE' column to datetime
acquisition_data['ANNOUNCED_DATE'] = pd.to_datetime(acquisition_data['ANNOUNCED_DATE'])

# Extract the year from 'ANNOUNCED_DATE'
acquisition_data['Year'] = acquisition_data['ANNOUNCED_DATE'].dt.year

# Round the years to integers
acquisition_data['Year'] = acquisition_data['Year'].astype(int).astype(str)

# Group data by 'Year' and 'ACQUISITION_TYPE'
grouped_data = acquisition_data.groupby(['Year', 'ACQUISITION_TYPE']).size().unstack(fill_value=0)

# Plot the graph
plt.figure(figsize=(12, 8))
for column in grouped_data.columns:
    plt.plot(grouped_data.index, grouped_data[column], label=column)

# Set labels and title
plt.xlabel('Year')
plt.ylabel('Quantity')
plt.title('Distribution of Acquisition Types Over Years')
plt.legend()

# Show the graph
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### *Employee Table*
```python
# Show basic information of the EMPLOYEE sheet
print("\nSheet: 04. EMPLOYEE")

# Descriptive statistics
employee_description = employee_data.describe().transpose()
employee_description['Data Type'] = employee_data.dtypes

# Convert the description to a structured table
descriptive_table = tabulate(employee_description, headers='keys', tablefmt='pretty')

# Null values
null_counts = employee_data.isnull().sum()

# Create a summary table for null values
null_summary_table = []
for column, null_count in null_counts.items():
    null_summary_table.append([column, null_count])

# Convert the summary table to a structured table
null_table = tabulate(null_summary_table, headers=['Variable', 'Null Count'], tablefmt='pretty')

# Print tables
print(descriptive_table)

# Number of variables
print("Number of Variables:", len(employee_data.columns))

# Number of records
print("Number of Records:", len(employee_data))

print("\nNull Values:")
print(null_table)
```
```python
# Show basic information of the NEWS sheet
print("\nSheet: 05. NEWS")

# Descriptive statistics
news_description = news_data.describe().transpose()

# Convert the description to a structured table
descriptive_table = tabulate(news_description, headers='keys', tablefmt='pretty')

# Null values
null_counts = news_data.isnull().sum()

# Create a summary table for null values with data type
null_summary_table = []
for column, null_count in null_counts.items():
    data_type = news_data[column].dtype
    null_summary_table.append([column, null_count, data_type])

# Convert the summary table to a structured table
null_table = tabulate(null_summary_table, headers=['Variable', 'Null Count', 'Data Type'], tablefmt='pretty')

# Print both tables
print(descriptive_table)

# Number of variables
print("Number of Variables:", len(news_data.columns))

# Number of records
print("Number of Records:", len(news_data))

print("\nNull Values:")
print(null_table)
```
```python
# Convert 'NEWS_DATE' column to datetime
news_data['NEWS_DATE'] = pd.to_datetime(news_data['NEWS_DATE'])

# Extract year from 'NEWS_DATE'
news_data['Year'] = news_data['NEWS_DATE'].dt.year

# Count number of news articles for each year
news_per_year = news_data['Year'].value_counts().sort_index()

# Plot
plt.figure(figsize=(10, 6))
plt.plot(news_per_year.index, news_per_year.values, marker='o', linestyle='-')
plt.title('Quantity of News per Year')
plt.xlabel('Year')
plt.ylabel('Number of News Articles')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
```

# Data Transformation and Cleaning
*(The file is also available in the repository: Data Cleaning & Transformation - Group G.ipynb)*

Once we've completed the data exploration, we move on to the transformation and cleaning of each table in order to prepare for modeling.

We install necesary libraries
```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter, defaultdict
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
!pip install shap
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# Connection to Drive
from google.colab import drive
drive.mount('/content/drive')
```

Assign each data set to a table:
```python
company_data = pd.read_excel('/content/drive/MyDrive/Applied Data Science Research Project/Colab Code/Data_modified.xlsx', sheet_name='01.COMPANY')
investment_data = pd.read_excel('/content/drive/MyDrive/Applied Data Science Research Project/Colab Code/Data_modified.xlsx', sheet_name='02.INVESTMENT')
acquisition_data = pd.read_excel('/content/drive/MyDrive/Applied Data Science Research Project/Colab Code/Data_modified.xlsx', sheet_name='03.ACQUISITION')
employee_data = pd.read_excel('/content/drive/MyDrive/Applied Data Science Research Project/Colab Code/Data_modified.xlsx', sheet_name='04.EMPLOYEE')
news_data_classification = pd.read_excel('/content/drive/MyDrive/Applied Data Science Research Project/Colab Code/Data_modified.xlsx', sheet_name='News with classification')
```

### *Company Table*

For clarification:

![image](https://github.com/Depascuali/Predicting-StartUp-Success-ML/assets/97790973/92c4c325-718a-47d8-9347-cfc2687d865a)

```python
# Divide column 'CATEGORY' in many columns
categories_expanded = company_data['CATEGORY'].str.split(', ', expand=True)

# Rename new columns
categories_expanded.columns = [f'category {i+1}' for i in range(categories_expanded.shape[1])]

# Drop original 'CATEGORY' column
company_data.drop('CATEGORY', axis=1, inplace=True)

# Concat original DataFrame (without column 'CATEGORY') with new categories data frame
company_data = pd.concat([company_data, categories_expanded], axis=1)

print(company_data)
```
```python
# Divide column 'LOCATION' en 4 segments, as some rows have 4 segments
location_components = company_data['LOCATION'].str.split(', ', expand=True)

# When there are 4 segments, there is an additional segment that can be ignored
company_data['city'] = location_components[0]
company_data['state'] = location_components[1]

# 'country'assignation depends if there is a forth segment (which  would indicate a country)
company_data['country'] = location_components[3].where(location_components[3].notnull(), location_components[2])

# Print result to verify
print(company_data[['COMPANY_NAME', 'city', 'state', 'country']])
```
```python
# Transform dates in years to be the first day of that year
def convert_year_first_day(series):
    # Convert values that appear to be just a year to 'yyyy-01-01'
    series = series.apply(lambda x: f"{x}-01-01" if (isinstance(x, int) or (isinstance(x, str) and x.isdigit())) else x)
    # Now try to convert all entries to datetime
    return pd.to_datetime(series, errors='coerce')

# Apply the function to each relevant column
company_data['FOUNDED_ON'] = convert_year_first_day(company_data['FOUNDED_ON'])
company_data['EXITED_ON'] = convert_year_first_day(company_data['EXITED_ON'])
company_data['CLOSED_ON'] = convert_year_first_day(company_data['CLOSED_ON'])


# Show some conversions to verify
print(company_data[['FOUNDED_ON', 'EXITED_ON', 'CLOSED_ON']].head())
```
```python
file_path = '/content/drive/My Drive/company_.xlsx'

# Export Data Frame
company_data.to_excel(file_path, index=False)
```

### *Investment Table*

For clarification:

![image](https://github.com/Depascuali/Predicting-StartUp-Success-ML/assets/97790973/c29892ea-b27a-4ce5-8c27-705b914ceae6)


```python
# Convert date columns to datetime format
investment_data['ANNOUNCED_DATE'] = pd.to_datetime(investment_data['ANNOUNCED_DATE'])
```

Create "ADVANCED_STAGE" field to determine if a company reached an advanced investment round
```python
investment_data['ADVANCED_STAGE'] = investment_data['INVESTMENT_STAGE'].apply(lambda x: 'No' if x == 'Seed' else 'Yes')
```
```python
# Function to determine the preferred row in case of multiple entries
def elegir_fila_grupo(grupo):
    if 'Yes' in grupo['ADVANCED_STAGE'].values:
        grupo_filtrado = grupo[grupo['ADVANCED_STAGE'] == 'Yes']
    else:
        grupo_filtrado = grupo  # If all are 'No', keep the group as is

    # Select the row with the earliest date
    return grupo_filtrado.sort_values(by='ANNOUNCED_DATE').iloc[0]

# Group by COMPANY_ID and apply the function to select the desired row
investment_data = investment_data.groupby('COMPANY_ID').apply(elegir_fila_grupo).reset_index(drop=True)

# Print Result
print(investment_data)
```
```python
# Create a list of unique COMPANY_IDs from the second table (investment_data)
company_ids_con_inversion = investment_data['COMPANY_ID'].unique().tolist()

# Mark with "Yes" all companies that had investment, with "No" those that never had
def determinar_avance(row):
    if row['COMPANY_ID'] in company_ids_con_inversion:
        return "Yes"
    else:
        return "No"

# Apply the function to the companies DataFrame to create the new column
company_data['ADVANCED_STAGE'] = company_data.apply(determinar_avance, axis=1)
```
```python
file_path = '/content/drive/My Drive/investment_data.xlsx'

# Export Data Frame
investment_data.to_excel(file_path, index=False)
```

### *Acquisition Table*

For clarification:

![image](https://github.com/Depascuali/Predicting-StartUp-Success-ML/assets/97790973/b5a66e5e-c79a-4020-a235-5f66839515ad)


```python
# Convert the 'ANNOUNCED_DATE' column from string to datetime to correctly sort the dates
investment_data['ANNOUNCED_DATE'] = pd.to_datetime(investment_data['ANNOUNCED_DATE'], errors='coerce')

# Filter to exclude entries where INVESTMENT_STAGE is "Seed"
filtered_datas = investment_data[investment_data['INVESTMENT_STAGE'] != 'Seed']

# Sort the filtered DataFrame by 'COMPANY_ID' and 'ANNOUNCED_DATE' (ascending)
sorted_filtered_datas = filtered_datas.sort_values(by=['COMPANY_ID', 'ANNOUNCED_DATE'])

# Keep only the oldest row for each 'COMPANY_ID'
unique_company_rowz = sorted_filtered_datas.drop_duplicates(subset='COMPANY_ID', keep='first')

# Print Result
print(unique_company_rowz)
```
```python
# Join Data to compare acquisitions previous to reaching an investment round
joined_data = pd.merge(
    acquisition_data[['ACQUIRER_ID', 'ANNOUNCED_DATE']],
    unique_company_rowz[['COMPANY_ID', 'ANNOUNCED_DATE', 'INVESTMENT_STAGE']],
    left_on='ACQUIRER_ID',
    right_on='COMPANY_ID',
    how='right',
    suffixes=('_ACQUISITION', '_INVESTMENT')
)
```
```python
# Ensure the ANNOUNCED_DATE_ACQUISITION and ANNOUNCED_DATE_INVESTMENT columns are in datetime format
joined_data['ANNOUNCED_DATE_ACQUISITION'] = pd.to_datetime(joined_data['ANNOUNCED_DATE_ACQUISITION'])
joined_data['ANNOUNCED_DATE_INVESTMENT'] = pd.to_datetime(joined_data['ANNOUNCED_DATE_INVESTMENT'])

# Filter the DataFrame where ANNOUNCED_DATE_ACQUISITION is before ANNOUNCED_DATE_INVESTMENT
filtered_data = joined_data[joined_data['ANNOUNCED_DATE_ACQUISITION'] < joined_data['ANNOUNCED_DATE_INVESTMENT']]

# Display unique ACQUIRER_IDs from the filtered data
unique_acquirer_ids = filtered_data['ACQUIRER_ID'].unique()
print(unique_acquirer_ids)
```
```python
number_of_unique_acquirers = len(unique_acquirer_ids)
print(number_of_unique_acquirers)
```
As there are 136 companies that made an acquisition before reaching a late stage investment, this analysis would not contribute the model. 136/9971 = 1.4%

### *Employee Table*
The objective is to create a new column to determine if a company has employees that studied in the top 20 universities in the world.

For clarification:

![image](https://github.com/Depascuali/Predicting-StartUp-Success-ML/assets/97790973/a22e23d2-3cad-4506-a6b1-0d8a46a8ee32)


```python
Top20_Universities = [
    "University of Oxford",
    "Oxford University",
    "MIT",
    "Stanford University",
    "Stanford Graduate School of Business",
    "Stanford",
    "Massachusetts Institute of Technology (MIT)",
    "Massachusetts Institute of Technology - MIT",
    "MIT",
    "Harvard University",
    "Harvard",
    "Cambridge College",
    "University of Cambridge",
    "Princeton University",
    "Caltech",
    "California Institute of Technology",
    "California Institute of Technology (Caltech)",
    "Imperial College London",
    "University of California, Berkeley",
    "University of California, Berkeley (UCB)",
    "UC Berkeley",
    "Yale University",
    "Yale",
    "ETH Zurich",
    "Tsinghua University",
    "Tsing Hua",
    "The University of Chicago",
    "University of Chicago",
    "Peking University",
    "Johns Hopkins University",
    "Johns Hopkins",
    "University of Pennsylvania",
    "Columbia University",
    "Columbia Business School",
    "University of California, Los Angeles (UCLA)",
    "UCLA",
    "University of California",
    "National University of Singapore (NUS)",
    "National University of Singapore",
    "Cornell University"
]

#Preprocessing the list of universities to ignore uppercase and spaces
Top20_Universities = [university.lower().replace(" ", "") for university in Top20_Universities]

# Initialize a new column in the DataFrame to indicate the presence of a Top 20 University
employee_data['Top20_University_Present'] = 'No'

# Function to check for the presence of any university from the list in the ATTENDED_SCHOOLS column
def check_top20_university(row):
    attended_schools = str(row['ATTENDED_SCHOOLS']).lower().replace(" ", "")
    for university in Top20_Universities:
        if university in attended_schools:  # Case-insensitive and spaceless search
            return 'Yes'
    return 'No'

# Apply the function to each row
employee_data['Top20_University_Present'] = employee_data.apply(check_top20_university, axis=1)

# Show the first rows of the DataFrame to verify the result
print(employee_data[['ATTENDED_SCHOOLS', 'Top20_University_Present']].head())
```
```python
# Split the 'COMPANY_IDS' column by commas and then use explode to expand the resulting lists into rows
employee_data['COMPANY_IDS'] = employee_data['COMPANY_IDS'].str.split(',')
employee_data = employee_data.explode('COMPANY_IDS')

# Reset the index of the resulting DataFrame to clean up the index after explode
employee_data.reset_index(drop=True, inplace=True)

# Rename Column
employee_data.rename(columns={'COMPANY_IDS': 'COMPANY_ID'}, inplace=True)
```

### *News Data Table*

For clarification:

![image](https://github.com/Depascuali/Predicting-StartUp-Success-ML/assets/97790973/5487145a-c2e2-4a07-b2df-8fc386f427e9)


In this table, we first run locally the sentiment analysis (it was impossible to do it in Collab as it would take a lot of GPU and time. We obtained a result table with the classifications for each news and used this as our base News Data Table. So, first the Visual Studio Code:
*(The file is also available in the repository: Sentiment Analysis - Hugging Face - Group G)*

```python
import pandas as pd
from transformers import pipeline
import re

# Path for Excel files
company_df_path = r'C:\Users\tomas\Desktop\Data Science Research Project\2324RX19_Project_Data.xlsx'
news_df_path = r'C:\Users\tomas\Desktop\Data Science Research Project\2324RX19_Project_Data.xlsx'

# Load the data frames
company_df = pd.read_excel(company_df_path, sheet_name='01.COMPANY')
news_df = pd.read_excel(news_df_path, sheet_name='05.NEWS')

# Merge the data frames
merged_df = pd.merge(news_df, company_df, on='COMPANY_ID', how='left')

# Filter the dataframe for headlines containing the company name
filtered_df = merged_df[merged_df.apply(lambda x: str(x['COMPANY_NAME']).lower().replace(' ', '') in str(x['NEWS_TITLE']).lower().replace(' ', ''), axis=1)]

# Function to replace specific sequence and characters around it with a single quote
def replace_sequence_with_quote(text):
    # The regular expression looks for 3 characters of any kind to the left of 'x0080' and 8 characters of any kind to the right.
    pattern = r'.{2}Â.{5}'
    # Replace pattern matches with a single quote
    replaced_text = re.sub(pattern, "", text)
    return replaced_text

# Apply the replacement function to the NEWS_TITLE column
filtered_df['NEWS_TITLE'] = filtered_df['NEWS_TITLE'].apply(replace_sequence_with_quote)

# Initialize the sentiment analysis pipeline with the specific model
classifier = pipeline('text-classification', model='mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis')

# Function to classify the sentiment of news headlines in batches, including the company name
def classify_sentiments_in_batches(titles, company_names, batch_size=32):
    results = []
    for i in range(0, len(titles), batch_size):
        batch_titles = titles[i:i + batch_size]
        batch_company_names = company_names[i:i + batch_size]
        batch_texts = [f"{company_name} - {title}" for company_name, title in zip(batch_company_names, batch_titles)]
        batch_results = classifier(batch_texts)
        results.extend([result['label'] for result in batch_results])
    return results

# Apply batch sentiment analysis and add the results to the DataFrame
filtered_df['Clasificacion'] = classify_sentiments_in_batches(filtered_df['NEWS_TITLE'].tolist(), filtered_df['COMPANY_NAME'].tolist())

# Save output file
output_path = r'C:\Users\tomas\Desktop\Data Science Research Project\dataset_con_clasificaciones_news_model_considering_company.csv'
filtered_df.to_csv(output_path, index=False)
```
The provided code allowed us to classify each news title taking in consideration the company name. We used Hugging Face API and the model: mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis. Link: https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis
We considered it was perfect for the use case.

We now continue with the data transformation having the resultant Visual Studio table as base:

```python
news_data_classification.info()
```
```python
# Change Format to DateTime
news_data_classification['NEWS_DATE'] = pd.to_datetime(news_data_classification['NEWS_DATE'], errors='coerce')

# Show some of the conversions to verify
print(news_data_classification[['NEWS_DATE']].head())
```
```python
# Drop unnecessary columns
news_data_classification = news_data_classification.drop(columns=['COMPANY_NAME', 'CATEGORY', 'LOCATION', 'FOUNDED_ON', 'EXITED_ON', 'CLOSED_ON'])

# Show the first rows of the resulting DataFrame to verify
print(news_data_classification.head())
```

# Creation of final table

![image](https://github.com/Depascuali/Predicting-StartUp-Success-ML/assets/97790973/f600bcc2-f1f7-4557-bb49-b6c5e2503bd0)


We now have applied the cleaning and transformation to every table, now it's time to create the final data set.
```python
def contar_noticias(row, news_data_classification):
    # Filter news by COMPANY_ID and by date before ANNOUNCED_DATE
    noticias_previas = news_data_classification[(news_data_classification['COMPANY_ID'] == row['COMPANY_ID']) & (news_data_classification['NEWS_DATE'] < row['ANNOUNCED_DATE'])]

    # Count classifications
    conteo = noticias_previas['Clasificacion'].value_counts().to_dict()

    # Return count for each classification, 0 if there are no news
    row['Positive'] = conteo.get('positive', 0)
    row['Negative'] = conteo.get('negative', 0)
    row['Neutral'] = conteo.get('neutral', 0)

    return row

# Apply the function to each row in the investment DataFrame
df_investment_news = investment_data.apply(lambda row: contar_noticias(row, news_data_classification), axis=1)
```
```python
# Filter companies that didn't receive investment
company_no_investment = company_data[company_data['ADVANCED_STAGE'] == 'No']
```
```python
# Change Data Type to integer
news_data_classification['Positive'] = (news_data_classification['Clasificacion'] == 'positive').astype(int)
news_data_classification['Negative'] = (news_data_classification['Clasificacion'] == 'negative').astype(int)
news_data_classification['Neutral'] = (news_data_classification['Clasificacion'] == 'neutral').astype(int)

# Group by COMPANY_ID and sum
df_news_counts = news_data_classification.groupby('COMPANY_ID', as_index=False).agg({
    'Positive': 'sum',
    'Negative': 'sum',
    'Neutral': 'sum'
})

# Step 2: Merge df_company with df_news_counts
# Merge (left join) df_company with df_news_counts on 'COMPANY_ID'
company_no_investment_news = pd.merge(company_no_investment, df_news_counts, on='COMPANY_ID', how='left')

# Fill NaNs with 0s for the 'Positive', 'Negative', 'Neutral' columns after the merge
company_no_investment_news[['Positive', 'Negative', 'Neutral']] = company_no_investment_news[['Positive', 'Negative', 'Neutral']].fillna(0).astype(int)
```
```python
# Merge Investment and Company data sets
df_merged = pd.merge(df_investment_news, company_data, on='COMPANY_ID', how='inner')
```
```python
# Drop the column 'ADVANCED_STAGE_y'
df_merged =  df_merged.drop(columns=['ADVANCED_STAGE_y'])

# Rename 'ADVANCED_STAGE_x' to 'ADVANCED
df_merged = df_merged.rename(columns={'ADVANCED_STAGE_x': 'ADVANCED_STAGE'})
```
```python
# Build Final DF by concat the companies that received Investment with their classification and the ones that didn't
final_df = pd.concat([df_merged, company_no_investment_news], ignore_index=True)
```
```python
# Drop specified columns from final_df
final_df = final_df.drop(columns=['FUNDING_TYPE', 'ANNOUNCED_DATE', 'INVESTMENT_STAGE','FOUNDED_ON', 'EXITED_ON', 'CLOSED_ON'])

# Show DataFrame
print(final_df)
```

```python
# Join Final DF with Employees data
final_df = pd.merge(final_df, employee_data, on='COMPANY_ID', how='left')
```
```python
# Drop the 'JOB_TITLES' and 'ATTENDED_SCHOOLS' columns from final_df
final_df = final_df.drop(['JOB_TITLES', 'ATTENDED_SCHOOLS'], axis=1)

# Fill blank/missing values in 'Top20_University_Present' with 'No'
final_df['Top20_University_Present'] = final_df['Top20_University_Present'].fillna('No')
```
```python
# Keep only an unique row, as the join repetead rows
final_df = final_df.drop_duplicates(subset='COMPANY_ID', keep='first')
```

Using GPT, we classified industries in broader classifications to further segment the data. 
```python
# Define broader industry categories
industries = {
    "Technology": ["3D Printing", "3D Technology", "A/B Testing", "Ad Exchange", "Ad Network", "Ad Retargeting", "Ad Server",
    "Ad Targeting", "Advanced Materials", "Analytics", "Android", "Angel Investment", "App Discovery",
    "App Marketing", "Application Performance Management", "Application Specific Integrated Circuit (ASIC)", "Apps",
    "Artificial Intelligence", "Augmented Reality", "Auto Insurance", "Automotive", "Autonomous Vehicles", "B2B",
    "B2C", "Battery", "Big Data", "Billing", "Bioinformatics", "Blockchain", "Blogging Platforms", "Browser Extensions",
    "Business Information Systems", "Business Intelligence", "CAD", "Call Center", "Cloud Computing", "Cloud Data Services","Cloud Infrastructure", "Cloud Management", "Cloud Security", "Cloud Storage", "CMS",
    "Collaboration", "Collaborative Consumption", "Communication Hardware", "Communications Infrastructure",
    "Computer", "Computer Vision", "Content", "Content Creators", "Content Delivery Network",
    "Content Discovery", "Content Marketing", "Content Syndication", "CRM", "Crowdfunding",
    "Crowdsourcing", "Cryptocurrency", "Cyber Security", "Data Center", "Data Center Automation",
    "Data Integration", "Data Mining", "Data Storage", "Data Visualization", "Database",
    "Developer APIs", "Developer Platform", "Developer Tools", "Digital Entertainment",
    "Digital Marketing", "Digital Media", "Digital Signage", "Direct Marketing", "Direct Sales",
    "Document Management", "Document Preparation", "Domain Registrar", "DRM", "Drone Management",
    "Drones", "DSP", "E-Commerce", "E-Commerce Platforms", "E-Learning", "E-Signature",
    "EBooks", "Ediscovery", "Electric Vehicle", "Electrical Distribution", "Electronic Design Automation (EDA)", "Electronic Health Record (EHR)",
    "Electronics", "Email", "Email Marketing", "Embedded Software", "Embedded Systems", "Emergency Medicine",
    "Emerging Markets", "Energy Efficiency", "Energy Management", "Energy Storage", "Enterprise", "Enterprise Applications",
    "Enterprise Resource Planning (ERP)", "Enterprise Software", "Environmental Consulting", "Environmental Engineering",
    "eSports", "Ethereum", "Event Management", "Event Promotion", "Events", "Eyewear", "Facebook", "Facial Recognition",
    "Facilities Support Services", "Facility Management", "Fantasy Sports", "Field Support", "Field-Programmable Gate Array (FPGA)",
    "File Sharing", "Film", "Film Distribution", "Film Production", "Flash Storage", "Fleet Management", "Fraud Detection",
    "Freelance", "Freemium", "Freight Service", "Fuel Cell", "Funding Platform", "Gambling", "Gamification", "Gaming",
    "Generation Z", "Genetics", "Geospatial", "Google", "Google Glass", "Government", "GovTech", "GPS", "GPU", "Graphic Design",
    "Green Building", "Green Consumer Goods", "GreenTech", "Grocery", "Group Buying", "Guides", "Handmade", "Hardware","Human Computer Interaction", "IaaS", "Identity Management", "Image Recognition",
    "Indoor Positioning", "Information and Communications Technology (ICT)", "Information Services",
    "Information Technology", "Infrastructure", "Innovation Management", "InsurTech", "Intelligent Systems",
    "Internet", "Internet of Things", "Internet Radio", "Intrusion Detection", "iOS", "ISP",
    "IT Infrastructure", "IT Management", "Machine Learning", "Management Information Systems","Messaging", "mHealth", "Mobile", "Mobile Advertising", "Mobile Apps", "Mobile Devices", "Mobile Payments",
    "MOOC", "Motion Capture", "Music Streaming", "Nanotechnology", "Natural Language Processing", "Navigation",
    "Network Hardware", "Network Security", "Neuroscience", "NFC", "Online Auctions", "Online Forums", "Online Games",
    "Online Portals", "Open Source", "Operating Systems", "Optical Communication", "Outsourcing", "PaaS",
    "Penetration Testing", "Personalization", "Photo Editing", "Photo Sharing", "Photography", "Physical Security",
    "Podcast", "Point of Sale", "Pollution Control", "Power Grid", "Predictive Analytics", "Presentation Software",
    "Privacy", "Private Cloud", "Private Social Networking", "Product Design", "Product Management","Product Search", "Productivity Tools", "Professional Networking", "Project Management",
    "Property Management", "Public Relations", "Public Safety", "Public Transportation", "Publishing", "Q&A",
    "QR Codes", "Quality Assurance", "Quantified Self", "Quantum Computing", "Reading Apps", "Real Time",
    "Recruiting", "Recycling", "Renewable Energy", "Reservation", "RFID", "Ride Sharing", "Risk Management",
    "Robotics", "SaaS", "Sales", "Sales Automation", "Same Day Delivery", "Satellite Communication",
    "Scheduling", "Search Engine", "Security", "Self-Storage", "SEM", "Semantic Search", "Semantic Web",
    "Semiconductor", "Sensor", "SEO", "Service Industry", "Sharing Economy", "Shipping", "Shipping Broker",
    "Shopping", "Shopping Mall", "Simulation", "Skill Assessment", "Smart Building", "Smart Cities",
    "Smart Home", "SMS", "Social", "Social Bookmarking", "Social CRM", "Social Entrepreneurship",
    "Social Impact", "Social Media", "Social Media Advertising", "Social Media Management", "Social Media Marketing",
    "Social Network", "Social News", "Social Recruiting", "Social Shopping", "Software", "Software Engineering","Solar", "Space Travel", "Speech Recognition", "Subscription Service", "Supply Chain Management",
    "Task Management", "Technical Support", "Telecommunications", "Test and Measurement", "Text Analytics",
    "Ticketing", "Translation Service", "Transportation", "Travel Accommodations", "Travel Agency", "Tutoring",
    "TV", "TV Production", "Unified Communications", "UX Design", "Vacation Rental", "Venture Capital",
    "Vertical Search", "Video", "Video Advertising", "Video Chat", "Video Conferencing", "Video Editing",
    "Video Games", "Video on Demand", "Video Streaming", "Virtual Assistant", "Virtual Currency",
    "Virtual Desktop", "Virtual Goods", "Virtual Reality", "Virtual Workforce", "Virtual World", "Virtualization",
    "Visual Search", "VoIP", "Warehousing", "Waste Management", "Water Purification", "Web Apps",
    "Web Browsers", "Web Design", "Web Development", "Web Hosting"],

    "Health & Wellness": ["Alternative Medicine", "Assisted Living", "Assistive Technology", "Biopharma", "Biotechnology", "Clinical Trials","Cosmetic Surgery", "Cosmetics", "Dental", "Dietary Supplements", "Diabetes","Elder Care", "Elderly", "Emergency Medicine", "Fertility","Health Care", "Health Diagnostics", "Health Insurance", "Home Health Care", "Horticulture",
    "Hospital", "Humanitarian", "Medical", "Medical Device", "mHealth", "Military", "Mineral", "Nutrition", "Nursing and Residential Care",
    "Nutrition", "Outpatient Care", "Personal Health", "Pharmaceutical","Psychology", "Rehabilitation","Wellness", "Therapeutics"],

    "Finance": ["Accounting", "Angel Investment", "Asset Management", "Banking", "Bitcoin","Credit", "Credit Bureau", "Credit Cards", "Commercial Insurance", "Commercial Lending",  "Finance", "Financial Exchanges", "Financial Services", "FinTech","Micro Lending", "Mobile Payments", "Personal Finance", "Prediction Markets","Property Insurance", "Real Estate", "Real Estate Investment", "Retail","Stock Exchanges", "Wealth Management"],

    "Education": ["E-Learning", "Continuing Education", "Corporate Training", "E-Learning", "Ediscovery","EdTech", "Education", "Edutainment", "Higher Education", "Language Learning","Meeting Software", "Men's", "MOOC", "Music Education", "Primary Education","Professional Services", "Secondary Education","STEM Education", "Universities", "Vocational Education"],

    "Manufacturing": ["Advanced Materials", "Aerospace", "Agriculture", "AgTech", "Automotive", "Battery", "Biotechnology",
    "Chemical", "Chemical Engineering", "Civil Engineering", "Clean Energy", "CleanTech","Food Processing", "Forestry", "Fossil Fuels", "Furniture","Industrial", "Industrial Automation", "Industrial Design", "Industrial Engineering",
    "Industrial Manufacturing", "Machinery Manufacturing", "Manufacturing","Mechanical Engineering", "Medical Device", "Mining", "Mining Technology", "Mobile Devices", "Musical Instruments",
    "Plastics and Rubber Manufacturing","Textiles", "Toys","Energy"],

    "Miscellaneous": ["Adult", "Adventure Travel", "Advertising", "Advertising Platforms", "Advice", "Air Transportation", "Alumni",
    "American Football", "Animal Feed", "Animation", "Aquaculture", "Architecture", "Art", "Association", "Auctions",
    "Audio", "Baby", "Bakery", "Basketball", "Beauty", "Biofuel", "Biomass Energy", "Biometrics", "Boating",
    "Brand Marketing", "Brewing", "Broadcasting", "Building Maintenance", "Building Material", "Business Development",
    "Business Travel", "Cannabis", "Car Sharing", "Career Planning", "Casino", "Casual Games", "Catering",
    "Cause Marketing", "Celebrity", "Charity", "Child Care", "Children", "CivicTech", "Classifieds","Coffee", "Collectibles", "College Recruiting", "Commercial", "Commercial Real Estate",
    "Communities", "Compliance", "Concerts", "Confectionery", "Console Games", "Construction",
    "Consulting", "Consumer", "Consumer Applications", "Consumer Electronics", "Consumer Goods",
    "Consumer Lending", "Consumer Research", "Consumer Reviews", "Consumer Software", "Contact Management",
    "Cooking", "Coupons", "Courier Service", "Coworking", "Craft Beer", "Creative Agency", "Cricket",
    "Cycling", "Delivery", "Delivery Service", "Family", "Farmers Market", "Farming", "Fashion", "Fast-Moving Consumer Goods", "Fuel", "Funerals",
    "Golf", "Gift", "Gift Card","Home and Garden", "Home Decor", "Home Improvement", "Home Renovation", "Home Services", "Homeland Security",
    "Homeless Shelter", "Hospitality", "Hotel", "Housekeeping Service", "Hunting", "Hydroponics", "Impact Investing",
    "In-Flight Entertainment", "Incubators", "Independent Music", "Insurance", "Intellectual Property", "Interior Design",
    "Janitorial Service", "Jewelry", "Journalism", "Knowledge Management", "Landscaping", "Laser", "Last Mile Transportation",
    "Laundry and Dry-cleaning", "Law Enforcement", "Lead Generation", "Lead Management", "Leasing", "Legal", "Legal Tech",
    "Leisure", "Lending", "LGBT", "Life Insurance", "Life Science", "Lifestyle", "Lighting", "Lingerie", "Linux", "Livestock",
    "Local", "Local Advertising", "Local Business", "Local Shopping", "Location Based Services", "Logistics", "Loyalty Programs",
    "macOS", "Made to Order", "Management Consulting", "Mapping Services", "Marine Technology", "Marine Transportation",
    "Market Research", "Marketing", "Marketing Automation", "Marketplace", "Mechanical Design","Media and Entertainment", "Millennials", "MMO Games", "Museums and Historical Sites", "Music", "Music Label",
    "Music Venues", "National Security", "Natural Resources", "News", "Nightclubs", "Nightlife", "Non Profit", "Nuclear",
    "Office Administration", "Oil and Gas", "Organic", "Organic Food", "Outdoor Advertising", "Outdoors", "Packaging Services",
    "Parenting", "Parking", "Payments", "PC Games", "Peer to Peer", "Performing Arts", "Personal Branding", "Personal Development",
    "Pet", "Politics", "Precious Metals", "Presentations", "Price Comparison", "Printing", "Procurement","Product Research", "Property Development", "Racing", "Railroad", "Recipes", "Recreation",
    "Recreational Vehicles", "Religion", "Rental", "Rental Property", "Reputation", "Residential",
    "Resorts", "Restaurants", "Retail Technology", "Retirement", "Seafood", "Serious Games",
    "Sex Industry", "Sex Tech", "Shoes", "Skiing", "Small and Medium Businesses", "Snack Food",
    "SNS", "Soccer", "Social","Sponsorship", "Sporting Goods", "Sports", "Staffing Agency", "Sustainability", "Swimming", "Taxi Service",
    "Tea", "Teenagers", "Timeshare", "Tobacco", "Tour Operator", "Tourism", "Trade Shows", "Trading Platform",
    "Training", "Transaction Processing", "Travel", "TV", "Vending and Concessions", "Veterinary",
    "Wedding", "Wholesale", "Wind Energy", "Wine And Spirits", "Winery", "Wired Telecommunications", "Wireless",
    "Women's", "Young Adults","Food and Beverage","Human Resources", "Water"]
}

# Function to group specific categories into broader industries
def categorize_industries(categories, industry_mapping):
    industry_groups = {}
    for industry, subcats in industry_mapping.items():
        for subcat in subcats:
            industry_groups[subcat] = industry
    return [industry_groups.get(category, "Other") for category in categories]
```
```python
# Invert the dictionary to facilitate the search
category_to_industry = {cat: industry for industry, cats in industries.items() for cat in cats}

# Prepare industry columns in the DataFrame
for industry in industries.keys():
    final_df[industry] = "No"  # Initialize all industries with "No"

# Function to update industry columns
def update_industry_columns(row):
    for i in range(1, 20):  # Assuming categories range from "category 1" to "category 19"
        category_col = f"category {i}"
        if pd.notna(row[category_col]):  # Check if there is a value in the category column
            industry = category_to_industry.get(row[category_col])
            if industry:
                row[industry] = "Yes"
    return row

# Apply function to each row
final_df = final_df.apply(update_industry_columns, axis=1)
```

This way, every row was now classified in 6 categories.

# Some interesting Analytics using Python and Tableau

```python
# Create DF to use for visualizations
final_df_for_analysis = final_df
final_df_for_analysis['Outcome'] = final_df['EXITED_ON'].apply(lambda x: 'Success' if isinstance(x, pd.Timestamp) else 'Non success')
final_df_for_analysis['Closed_Status'] = final_df['CLOSED_ON'].apply(lambda x: 'Closed' if isinstance(x, pd.Timestamp) else 'Not Closed')


print(final_df_for_analysis)
```
```python
# Group by ADVANCED_STAGE and sum the news counts
news_counts = final_df_for_analysis.groupby('ADVANCED_STAGE')[['Positive', 'Negative']].sum()

# Calculate percentages
total_news = news_counts.sum(axis=1)
news_percentages = news_counts.div(total_news, axis=0) * 100

# Plotting
ax = news_percentages.plot(kind='bar', figsize=(10, 6))
plt.title('Percentage of Positive and Negative News per ADVANCED_STAGE')
plt.xlabel('ADVANCED_STAGE')
plt.ylabel('Percentage')
plt.xticks(rotation=0)
plt.legend(title='News Type')

# Annotate percentages
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'{height:.1f}%', (x + width/2, y + height + 1), ha='center')

plt.show()
```
```python
# Group by Top20_University_Present and ADVANCED_STAGE, and count the number of companies
company_counts = final_df_for_analysis.groupby(['Top20_University_Present', 'ADVANCED_STAGE']).size().unstack(fill_value=0)

# Calculate percentages
company_percentages = company_counts.div(company_counts.sum(axis=1), axis=0) * 100

# Plotting
ax = company_percentages.plot(kind='bar', figsize=(10, 6))
plt.title('Percentage of Companies by Top20 University Presence and Advanced Stage')
plt.xlabel('Top20 University Present')
plt.ylabel('Percentage of Companies')
plt.xticks(rotation=0)
plt.legend(title='Advanced Stage')

# Annotate percentages
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'{height:.1f}%', (x + width/2, y + height + 1), ha='center')

plt.show()
```


![image](https://github.com/Depascuali/Predicting-StartUp-Success-ML/assets/97790973/15364d70-b036-496d-b3ef-709638b415d7)

On the first graph, we can see that Companies that didn't reach an advanced stage have 7.4% more negative news  previous to the investment round than companies that did reach an advance stage.

On the second graph, we observe that 70.4% of companies that had top management in Top 20 universities reached a late investment stage vs 58.1% in the case of companies that didn't had top management in Top 20 universities.
However, this might be explained by 2 reasons:
1) Most companies from the US are the most successful, and as most of the top 20 universities are as well from the US, there might be a strong correlation.
2) It's logic to think that companies with directors that studied on the Top 20 universites might have more contacts and resources to reach late investment stages.

Using Tableau:

![image](https://github.com/Depascuali/Predicting-StartUp-Success-ML/assets/97790973/31e7521e-b708-4dd5-b47b-816c8a17fed2)

Here, we can observe in green countries that have more Start Ups that reached an Advanced Investment Stage than companies that didn't. (Red is the other way round).
As wee can see, there is a problem with South American Start-Ups.


# Machine Learning Models

Having the merged Data Set, it looks as follow:

![image](https://github.com/Depascuali/Predicting-StartUp-Success-ML/assets/97790973/355bd733-448f-4030-a959-5675eeded4fd)

### Logitic Regression
```python
# Selecting features and target
X = final_df[['Positive', 'Negative', 'Neutral', 'city', 'state', 'country', 'Top20_University_Present', 'Technology','Health & Wellness','Finance','Education','Manufacturing','Miscellaneous']]
y = final_df['ADVANCED_STAGE']

# Preprocessing for categorical features
categorical_features = ['city', 'state', 'country', 'Top20_University_Present', 'Technology','Health & Wellness','Finance','Education','Manufacturing','Miscellaneous']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ])

# Create preprocessing and training pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', LogisticRegression(solver='liblinear'))])

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
model.fit(X_train, y_train)

# Predict on testing set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

### Rest of models (Random Forest, KNN, Gradient Boosting)
```python
# Selecting features and target
X = final_df[['Positive', 'Negative', 'Neutral', 'city', 'state', 'country', 'Top20_University_Present',
              'Technology', 'Health & Wellness', 'Finance', 'Education', 'Manufacturing', 'Miscellaneous']]
y = final_df['ADVANCED_STAGE']

# Encoding the target variable if it's categorical
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Since we have categorical features, let's use a simple approach to convert them to numeric
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# Make sure both training and test set have the same columns after encoding
X_train, X_test = X_train.align(X_test, join='inner', axis=1)  # This ensures both have the same columns

# Example model: Random Forest
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train, y_train)
rf_predictions = rf_clf.predict(X_test)
```
```python
# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_predictions)}")


#GRADIENT BOOSTING MACHINES
gbm_model = GradientBoostingClassifier(random_state=42)
gbm_model.fit(X_train, y_train)
gbm_predictions = gbm_model.predict(X_test)
print(f"Gradient Boosting Accuracy: {accuracy_score(y_test, gbm_predictions)}")


#KNN
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)
print(f"KNN Accuracy: {accuracy_score(y_test, knn_predictions)}")
```

The rest of the models (Bayes and SVM) where done in AzureML as follows:

![WhatsApp Image 2024-04-01 at 21 05 55_f9b6ef4b](https://github.com/Depascuali/Predicting-StartUp-Success-ML/assets/97790973/685e8e2a-78cf-432e-8ae8-5c05c17a6a58)

Now, we compare accuracy for the models:
```python
# Data for the models and their accuracy, unsorted
models = ['SVM', 'Logistic Regression', 'Gradient Boosting', 'Random Forest', 'Bayes', 'KNN']
accuracy = [57.0, 58.0, 57.6, 55.7, 57.1, 56.1]

# Combine models and accuracy into a list of tuples and sort by accuracy
model_accuracy_pairs = sorted(zip(models, accuracy), key=lambda x: x[1], reverse=True)

# Unzip the sorted pairs back into models and accuracy lists
sorted_models, sorted_accuracy = zip(*model_accuracy_pairs)

# Setting up the figure and axes for the bar graph
plt.figure(figsize=(12, 6))  # Adjust figure size for readability
bar_width = 0.6  # Increased bar width for less space between bars
bars = plt.bar(sorted_models, sorted_accuracy, color='#0c1b3f', width=bar_width)
plt.xlabel('Model')
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy Comparison - Sorted')
plt.xticks(rotation=45)  # Rotate the x-axis labels for readability
plt.ylim(50, 60)  # Extend y-axis limit for displaying values on top

# Loop through each bar to place the accuracy value on top
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f"{yval}%", ha='center', va='bottom')

# Adjust layout to ensure nothing overlaps and remove grid lines
plt.grid(False)
plt.tight_layout()
```

As we can see the results:

![image](https://github.com/Depascuali/Predicting-StartUp-Success-ML/assets/97790973/44cc1f16-d635-4f68-b1ab-638f3792e49c)

Logistic Regression had the best accuracy.

Further Anlysis on Logistic Regression:

```python
# Calculate the probabilities of the positive class
y_prob = model.predict_proba(X_test)[:, 1]

# Calculate the ROC curve points
fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label='Yes')  # Adjust pos_label based on your positive class

# Calculate the AUC (Area under the ROC Curve)
roc_auc = auc(fpr, tpr)

# Plotting the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
```

![image](https://github.com/Depascuali/Predicting-StartUp-Success-ML/assets/97790973/fc96a2ec-a000-420b-91ee-7c8949ab0b00)

```python
# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)
```
```python
# Assuming y_test and y_pred are defined
conf_matrix = confusion_matrix(y_test, y_pred)

# Plotting the confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)

# Labels, title and ticks
label_font = {'size':'16'}  # Adjust to fit
ax.set_xlabel('Predicted labels', fontdict=label_font);
ax.set_ylabel('True labels', fontdict=label_font);
ax.set_title('Confusion Matrix', fontdict={'size':'18'});
ax.tick_params(axis='both', which='major', labelsize=12)  # Adjust to fit
ax.xaxis.set_ticklabels(['Negative', 'Positive']); ax.yaxis.set_ticklabels(['Negative', 'Positive']);

plt.show()
```

![image](https://github.com/Depascuali/Predicting-StartUp-Success-ML/assets/97790973/3998e95e-763f-4324-ab38-ba2fa0071ea9)

Measuring Feature importance with SHAP value:
```python
# Prepare the data for SHAP (this requires X_train to be preprocessed)
X_train_preprocessed = model.named_steps['preprocessor'].transform(X_train)

# Initialize the SHAP explainer
explainer = shap.Explainer(model.named_steps['classifier'], X_train_preprocessed)

# Calculate SHAP values
shap_values = explainer.shap_values(X_train_preprocessed)
```
```python
# Compute the mean absolute SHAP values for each feature and keep only positive values
mean_shap_values_positive = np.maximum(np.mean(shap_values, axis=0), 0)

# Sort the features by their mean absolute SHAP values in descending order
sorted_indices = np.argsort(-mean_shap_values_positive)

# Select the top N features with the most positive impact
top_n = 10  # Adjust this to select the number of top features you want to display
top_indices = sorted_indices[:top_n]

# Create a summary plot with only the top positive impacting features
shap.summary_plot(shap_values[:, top_indices], X_train_preprocessed[:, top_indices],
                  feature_names=np.array(preprocessor.get_feature_names_out())[top_indices],
                  plot_type="bar")
```

Most positive impact:

![image](https://github.com/Depascuali/Predicting-StartUp-Success-ML/assets/97790973/9a71234f-06c9-4c39-aee8-39bf6d85f863)

```python
# Compute the mean SHAP values for each feature and keep only negative values
mean_shap_values_negative = np.minimum(np.mean(shap_values, axis=0), 0)

# Sort the features by their mean SHAP values in ascending order to get the most negative impact
sorted_indices = np.argsort(mean_shap_values_negative)

# Select the top N features with the most negative impact
top_n = 10  # Adjust this to select the number of top features you want to display
top_indices = sorted_indices[:top_n]

# Create a summary plot with only the top negative impacting features
shap.summary_plot(shap_values[:, top_indices], X_train_preprocessed[:, top_indices], feature_names=np.array(preprocessor.get_feature_names_out())[top_indices], plot_type="bar")
```

Most negative impact:

![image](https://github.com/Depascuali/Predicting-StartUp-Success-ML/assets/97790973/f1f03ff4-c405-4448-b83a-8440478165b2)




