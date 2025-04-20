#solve the five Objective on that dataset
#1. Identify Trends and Growth Rates:
#a) Calculate the daily increase in cases and deaths to understand the spread of the virus over time.
#b) Determine if certain racial groups experienced faster case growth rates compared to others.

#2. Assess Disparities:
# a) Compare crude case and death rates among racial groups to highlight disparities in COVID-19 exposure and outcomes.
#b) This compares the average crude case and death rates per 100k population across racial/ethnic groups.
#c) It includes visualizations and a t-test to assess statistically significant differences, e.g., between NH Black and NH White populations.

#3. Find Peak Case and Death Days:
#a) Identify the day with the highest increase in cases for each racial group.
#b) Identify the day with the highest increase in deaths.
# Pie Chart: Peak Total Cases per Racial Group and Peak Total Deaths per Racial Group.
# Detect outliers for Total Cases and Total Deaths.
# Boxplot for Total Cases and Total Deaths.

#4. Analyze Age-Adjusted Rates:
#a) Investigate why some racial groups have higher crude case rates but lower age-adjusted case rates (e.g., NH White population).
#b) Determine if age plays a significant role in the impact of COVID-19 across racial groups.

#5. Detect Data Gaps and Anomalies:
#a) Some rows have missing data for NH Multiracial and NH Other categories. Investigate why these categories have incomplete data.
#b) The "Unknown" category has a high number of total cases but a low number of deaths—this could indicate missing demographic data.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# Load the dataset (assuming a CSV file)
df = pd.read_csv(r"C:\Users\priya\OneDrive\Desktop\python\COVID-19_Cases_and_Deaths_by_Race_Ethnicity_-_ARCHIVE (1).csv")

# Display first few rows to understand the dataset
print("First 5 rows of the dataset:")
print(df.head())

# Data Cleaning

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())


# Handle missing values
df = df.ffill()  # Forward fill missing values
df.fillna(0, inplace=True)  # Replace remaining NaNs with 0

# Check for duplicates and remove them
if df.duplicated().sum() > 0:
    df.drop_duplicates(inplace=True)
    print("\nDuplicates removed.")
else:
    print("\nNo duplicate values found.")


# Convert date column to datetime format
df["Date updated"] = pd.to_datetime(df["Date updated"], errors='coerce')

# Ensure numeric columns are correctly formatted
numeric_cols = ["Total population", "Total cases", "Crude case rate per 100k", "Age adjusted case rate per 100k", "Total deaths", "Crude death rate per 100k", "Age adjusted death rate per 100k"]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Sorting the dataset by date
df.sort_values(by=["Date updated"], inplace=True)

# Exploratory Data Analysis (EDA)
print("Dataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())

# Summary statistics
print("\nSummary statistics of numerical columns:")
print(df.describe())

# Check unique values in categorical columns
if 'Race/ethnicity' in df.columns:
    print("\nUnique categories in Race/ethnicity column:")
    print(df['Race/ethnicity'].unique())

# Visualize missing values using a heatmap 
plt.figure(figsize=(10, 5))
sns.heatmap(df.isnull(), cmap="viridis", cbar=True)
plt.title("Missing Values Heatmap")
plt.show()

# Distribution of total cases per race
if 'Race/ethnicity' in df.columns and 'total_cases' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Race/ethnicity", y="total_cases", data=df)
    plt.xticks(rotation=45)
    plt.title("Total COVID-19 Cases by Race/Ethnicity")
    plt.show()

# Time-series analysis of cases (if date column exists)
if 'Date updated' in df.columns and 'total_cases' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.lineplot(x="Date updated", y="total_cases", hue="Race/ethnicity", data=df)
    plt.xticks(rotation=45)
    plt.title("COVID-19 Cases Over Time by Race/Ethnicity")
    plt.show()

# Feature Engineering

df["death_case_ratio"] = df["Total deaths"] / df["Total cases"]
borough_columns = [col for col in df.columns if "Hospitalized" in col]
df["total_hospitalized"] = df[borough_columns].sum(axis=1)
print(df[["death_case_ratio", "total_hospitalized"]].head())

# Creating a new column for case fatality rate (CFR)
if 'total_cases' in df.columns and 'total_deaths' in df.columns:
    df['case_fatality_rate'] = (df['total_deaths'] / df['total_cases']) * 100
    df['case_fatality_rate'] = df['case_fatality_rate'].fillna(0)  # Handle NaN if cases are zero
    print("\nCase Fatality Rate (CFR) added to dataset.")

# Correlation Analysis
plt.figure(figsize=(10, 6))

# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=[np.number])

sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Numeric Features")
plt.show()

# Save Cleaned Dataset 
df.to_csv("cleaned_data.csv", index=False)
print("\nCleaned dataset saved as 'cleaned_data.csv'.")

# Objective 1: Identify Trends and Growth Rates
# a) Calculate daily increase in cases and deaths
df['Daily case increase'] = df.groupby("Race/ethnicity")["Total cases"].diff().fillna(0)
df['Daily death increase'] = df.groupby("Race/ethnicity")["Total deaths"].diff().fillna(0)

# b) Determine if certain racial groups experienced faster case growth rates
race_growth_rates = df.groupby("Race/ethnicity")["Daily case increase"].mean().sort_values(ascending=False)
print("Average Daily Case Growth Rate by Race/Ethnicity:\n", race_growth_rates)

# Ensure 'Date updated' is in datetime format
df["Date updated"] = pd.to_datetime(df["Date updated"])

# Check if 'daily_new_cases' exists; if not, compute it
if "daily_new_cases" not in df.columns:
    df["daily_new_cases"] = df.groupby("Race/ethnicity")["Total cases"].diff().fillna(0)

# Plot daily new cases by race/ethnicity
plt.figure(figsize=(12, 6))
sns.lineplot(x="Date updated", y="daily_new_cases", hue="Race/ethnicity", data=df)
plt.xticks(rotation=45)
plt.title("Daily New COVID-19 Cases by Race/Ethnicity")
plt.show()

# Objective 2: Assess Disparities
# a) Compare case and death rates between racial groups
case_rate_comparison = df.groupby("Race/ethnicity")["Crude case rate per 100k"].mean()
death_rate_comparison = df.groupby("Race/ethnicity")["Crude death rate per 100k"].mean()

# b) Analyze age-adjusted death rates
age_adjusted_death_rates = df.groupby("Race/ethnicity")["Age adjusted death rate per 100k"].mean()
print("Age-Adjusted Death Rates by Race/Ethnicity:\n", age_adjusted_death_rates)

# Check and calculate case rate per 100k if missing
if "case_rate_per_100k" not in df.columns and "total_cases" in df.columns and "population" in df.columns:
    df["case_rate_per_100k"] = (df["total_cases"] / df["population"]) * 100000

# Check and calculate death rate per 100k if missing
if "death_rate_per_100k" not in df.columns and "total_deaths" in df.columns and "population" in df.columns:
    df["death_rate_per_100k"] = (df["total_deaths"] / df["population"]) * 100000

# Check if case_rate_per_100k exists before plotting
if "case_rate_per_100k" in df.columns:
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Race/ethnicity", y="case_rate_per_100k", data=df)
    plt.xticks(rotation=45)
    plt.title("COVID-19 Case Rates per 100K by Race/Ethnicity")
    plt.show()
else:
    print("Column 'case_rate_per_100k' not found in the dataset.")

# Check if death_rate_per_100k exists before plotting
if "death_rate_per_100k" in df.columns:
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Race/ethnicity", y="death_rate_per_100k", data=df)
    plt.xticks(rotation=45)
    plt.title("COVID-19 Death Rates per 100K by Race/Ethnicity")
    plt.show()
else:
    print("Column 'death_rate_per_100k' not found in the dataset.")

# Check if age-adjusted death rate exists before plotting
if "age_adjusted_death_rate_per_100000" in df.columns:
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Race/ethnicity", y="age_adjusted_death_rate_per_100000", data=df)
    plt.xticks(rotation=45)
    plt.title("Age-Adjusted Death Rates by Race/Ethnicity")
    plt.show()
else:
    print("Column 'age_adjusted_death_rate_per_100000' not found in the dataset.")

# T-Test – NH Black vs NH White Crude Case Rate
#Statistical test to determine if the crude case rate between NH Black and NH White groups is significantly different.
group1 = df[df['Race/ethnicity'] == 'NH Black']['Crude case rate per 100k']
group2 = df[df['Race/ethnicity'] == 'NH White']['Crude case rate per 100k']
t_stat, p_val = ttest_ind(group1, group2)
print("T-Statistic:", t_stat)
print("P-Value:", p_val)


# Objective 3: Find Peak Case & Death Days
# Check correct date column
date_col = "Date updated" 

#a) Identify the day with the highest increase in cases for each racial group.
peak_cases = df.loc[df.groupby("Race/ethnicity")["Total cases"].idxmax(), ["Race/ethnicity", date_col, "Total cases"]]
overall_peak_cases = df.loc[df["Total cases"].idxmax(), [date_col, "Total cases"]]

#b) Identify the day with the highest increase in deaths.
peak_deaths = df.loc[df.groupby("Race/ethnicity")["Total deaths"].idxmax(), ["Race/ethnicity", date_col, "Total deaths"]]
overall_peak_deaths = df.loc[df["Total deaths"].idxmax(), [date_col, "Total deaths"]]

# Print results
print("\nPeak COVID-19 Cases Per Racial Group:\n", peak_cases)
print("\nPeak COVID-19 Deaths Per Racial Group:\n", peak_deaths)
print("\nOverall Peak Cases Day:", overall_peak_cases)
print("Overall Peak Deaths Day:", overall_peak_deaths)

# Pie Chart: Peak Total Cases per Racial Group 
plt.figure(figsize=(8, 8))
plt.pie(peak_cases["Total cases"], labels=peak_cases["Race/ethnicity"], 
        autopct='%1.1f%%', startangle=140, colors=sns.color_palette("Blues"))
plt.title("Proportion of Peak COVID-19 Cases by Racial Group")
plt.axis('equal')  # Equal aspect ratio ensures pie is a circle.
plt.show()

# Pie Chart: Peak Total Deaths per Racial Group
plt.figure(figsize=(8, 8))
plt.pie(peak_deaths["Total deaths"], labels=peak_deaths["Race/ethnicity"], 
        autopct='%1.1f%%', startangle=140, colors=sns.color_palette("Reds"))
plt.title("Proportion of Peak COVID-19 Deaths by Racial Group")
plt.axis('equal')
plt.show()

# Outlier Detection Function
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# Detect outliers for Total Cases
case_outliers, case_lb, case_ub = detect_outliers_iqr(peak_cases, "Total cases")
print("\nOutliers in Peak Total Cases per Race/Ethnicity:\n", case_outliers)

# Detect outliers for Total Deaths
death_outliers, death_lb, death_ub = detect_outliers_iqr(peak_deaths, "Total deaths")
print("\nOutliers in Peak Total Deaths per Race/Ethnicity:\n", death_outliers)

# Boxplot for Total Cases
plt.figure(figsize=(8, 5))
sns.boxplot(data=peak_cases, x="Total cases")
plt.title("Boxplot of Peak Total Cases by Race/Ethnicity")
plt.show()

# Boxplot for Total Deaths
plt.figure(figsize=(8, 5))
sns.boxplot(data=peak_deaths, x="Total deaths")
plt.title("Boxplot of Peak Total Deaths by Race/Ethnicity")
plt.show()

# Objective 4: Analyze Age-Adjusted Rates
# a) Investigate disparities in crude vs. age-adjusted case rates
age_adjusted_case_rates = df.groupby("Race/ethnicity")["Age adjusted case rate per 100k"].mean()
disparities = case_rate_comparison - age_adjusted_case_rates
# b) Determine if age plays a role in COVID-19 impact
df['Age impact'] = df["Crude case rate per 100k"] - df["Age adjusted case rate per 100k"]
print("Impact of Age on COVID-19 Case Rates:\n", df.groupby("Race/ethnicity")["Age impact"].mean())
# Calculate rate_difference: Crude case rate - Age adjusted case rate
df["rate_difference"] = df["Crude case rate per 100k"] - df["Age adjusted case rate per 100k"]

# Filter racial groups with higher crude case rates but lower age-adjusted case rates
filtered_df = df[df["rate_difference"] > 0]
plt.figure(figsize=(12, 6))
sns.barplot(x="Race/ethnicity", y="rate_difference", data=filtered_df)
plt.xticks(rotation=45)
plt.title("Crude vs Age-Adjusted Case Rate Differences by Race")
plt.show()

# Objective 5: Detect Data Gaps and Anomalies
# a) Investigate missing data for NH Multiracial and NH Other
df_missing = df[df["Race/ethnicity"].isin(["NH Multiracial", "NH Other"])]
missing_counts = df_missing.isnull().sum()
print("Missing Data Counts:\n", missing_counts)

# b) Analyze the "Unknown" category for inconsistencies
unknown_category = df[df["Race/ethnicity"] == "Unknown"]
print("Unknown Category Analysis:\n", unknown_category[["Total cases", "Total deaths"]].describe())

# Filter the DataFrame for the 'Unknown' category
unknown_data = df[df["Race/ethnicity"] == "Unknown"]
plt.figure(figsize=(12, 6))

# Plot the line charts
sns.lineplot(x="Date updated", y="Total cases", data=unknown_data, label="Total Cases")
sns.lineplot(x="Date updated", y="Total deaths", data=unknown_data, label="Total Deaths")
plt.xticks(rotation=45)
plt.title("Cases vs Deaths for 'Unknown' Category")
plt.legend()
plt.show()
# Visualizations
plt.figure(figsize=(10, 5))
sns.barplot(x=race_growth_rates.index, y=race_growth_rates.values, hue=race_growth_rates.index, dodge=False, palette='viridis')
plt.title('Average Daily Case Growth Rate by Race/Ethnicity')
plt.xlabel('Race/Ethnicity')
plt.ylabel('Average Daily Cases')
plt.xticks(rotation=45)
plt.show()















