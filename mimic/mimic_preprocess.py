
import pandas as pd
import os
import numpy as np

patients = pd.read_csv(os.path.join( "PATIENTS.csv.gz" ), compression="gzip")
patients.columns = patients.columns.str.lower()
patients.head()


admissions = pd.read_csv(os.path.join("ADMISSIONS.csv.gz" ), compression="gzip")
admissions.columns = admissions.columns.str.lower()
admissions.head(5)

merged_df = pd.merge(admissions, patients, on='subject_id', how='inner')

merged_df=merged_df.drop(['dod','dod_hosp','dod_ssn','expire_flag','row_id_y','row_id_x','has_chartevents_data','deathtime'],axis=1)

merged_df.head()

def calculate_age(df):
    """
    Calculate age from admittime and dob with proper error handling
    """
    # Make a copy of the dataframe
    df = df.copy()

    # Convert to datetime
    df['admittime'] = pd.to_datetime(df['admittime'], errors='coerce')
    df['dob'] = pd.to_datetime(df['dob'], errors='coerce')

    # Drop rows where either admittime or dob is NaN
    df = df.dropna(subset=['admittime', 'dob'])

    # Filter rows where dob is after admittime
    df = df[df['dob'] <= df['admittime']]

    # Calculate age using a safer method
    def safe_age_calc(row):
        try:
            # Calculate age using datetime year attribute
            age = row['admittime'].year - row['dob'].year

            # Adjust age if birthday hasn't occurred yet that year
            if (row['admittime'].month, row['admittime'].day) < (row['dob'].month, row['dob'].day):
                age -= 1

            # Handle unrealistic ages
            if age < 0 or age > 120:
                return np.nan
            return age

        except Exception:
            return np.nan

    # Apply the safe age calculation
    df['age'] = df.apply(safe_age_calc, axis=1)

    # Handle missing and unrealistic ages
    median_age = df[df['age'].between(0, 120)]['age'].median()
    df['age'] = df['age'].fillna(median_age)

    # Ensure age is integer
    df['age'] = df['age'].astype(int)

    print("Age statistics:")
    print(df['age'].describe())

    return df

# Use the function
try:
    df = calculate_age(merged_df)
    print("\nAge calculation completed successfully")

    # Print age distribution
    print("\nAge distribution:")
    print(pd.cut(df['age'], bins=[0,18,30,50,70,90,120]).value_counts().sort_index())

except Exception as e:
    print(f"Error in age calculation: {str(e)}")

# Step 1: Convert datetime columns
datetime_columns = ['admittime', 'dischtime', 'dob', 'deathtime', 'dod', 'dod_hosp', 'dod_ssn']
for col in datetime_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

#     # Calculate LOS in days (if columns exist)
#     if 'admittime' in df.columns and 'dischtime' in df.columns:
#         df['los'] = (df['dischtime'] - df['admittime']).dt.total_seconds() / (24 * 60 * 60)
#         df['los'] = df['los'].round(2)
if 'admittime' in df.columns and 'dischtime' in df.columns:
    # Calculate LOS and handle any negative values
    df['los'] = (df['dischtime'] - df['admittime']).dt.total_seconds() / (24 * 60 * 60)
    # Replace negative values with 0 and round to 2 decimal places
    df['los'] = df['los'].apply(lambda x: max(0, x)).round(2)
else:
    df['los'] = 0


df['marital_status'].fillna('Unknown')

def categorize_los(los):
    if los <= 3:
        return 0
    elif 3 < los <= 7:
        return 1
    elif 7 < los <= 14:
        return 2
    else:  # los > 14
        return 3

# Create new column with categorical values
df['los'] = df['los'].apply(categorize_los)

# Simple replacement version
df['diagnosis'] = df['diagnosis'].astype(str).replace({r'\\': ',', '/': ','}, regex=True)

# Print results
print("\nSample of cleaned diagnoses:")
print(df['diagnosis'].head())

df.to_csv('mp_los_age_category.csv',index=False)



