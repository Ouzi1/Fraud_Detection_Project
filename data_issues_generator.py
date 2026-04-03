"""
Data Issues Generator - introduces realistic data quality problems for ML practice.

This module provides functions to intentionally introduce common data issues into
a clean dataset for practicing data cleaning and preprocessing techniques.

Typical usage:
    from data_issues_generator import introduce_data_issues
    df_corrupted = introduce_data_issues(df_clean, duplicate_pct=0.05)
"""

import pandas as pd
import numpy as np


def introduce_data_issues(df,
                         duplicate_pct=0.05,
                         missing_pct=0.05,
                         random_state=42):
    """
    Introduce data quality issues into the DataFrame for practice.

    Parameters
    ----------
    df : pandas.DataFrame
        The clean DataFrame to corrupt
    duplicate_pct : float, default=0.05
        Percentage of rows to duplicate (0.05 = 5%)
    missing_pct : float, default=0.05
        Percentage of missing values to introduce across features
    random_state : int, default=42
        Random seed for reproducibility

    Returns
    -------
    pandas.DataFrame
        DataFrame with introduced data issues

    Notes
    -----
    The following issues are introduced:

    1. Duplicates (5% by default)
       - Randomly samples existing rows and appends them

    2. Missing values (5% by default)
       - Distributed across 8 features: Gender, Account_Type, Transaction_Type,
         Merchant_Category, Transaction_Device, Device_Type, Age, Transaction_Amount

    3. Categorical inconsistencies in 3 fields:
       - Gender: lowercase, uppercase, abbreviations ('M', 'F')
       - Account_Type: mixed case, extra spaces, abbreviations
       - City: leading/trailing spaces, case variations

    4. Outliers in numerical features:
       - Age: 150, -5, 0, 999, 120, -1
       - Transaction_Amount: 1,000,000+, negative values
       - Account_Balance: 10,000,000+, negative values, extreme zeros
    """
    np.random.seed(random_state)
    df_issues = df.copy()
    n_rows = len(df_issues)

    # 1. Add duplicates
    n_duplicates = int(n_rows * duplicate_pct)
    duplicate_indices = np.random.choice(df_issues.index, size=n_duplicates, replace=True)
    duplicates = df_issues.iloc[duplicate_indices].copy()
    df_issues = pd.concat([df_issues, duplicates], ignore_index=True)

    # 2. Add missing values
    n_missing_total = int(n_rows * missing_pct)
    features_for_missing = [
        'Gender', 'Account_Type', 'Transaction_Type',
        'Merchant_Category', 'Transaction_Device',
        'Device_Type', 'Age', 'Transaction_Amount'
    ]
    missing_per_feature = max(1, n_missing_total // len(features_for_missing))

    for feature in features_for_missing:
        if feature in df_issues.columns:
            n_missing = min(missing_per_feature, len(df_issues))
            missing_indices = np.random.choice(df_issues.index, size=n_missing, replace=False)
            df_issues.loc[missing_indices, feature] = np.nan

    # 3. Data inconsistency in Gender
    if 'Gender' in df_issues.columns:
        n_inconsistent = int(len(df_issues) * 0.02)
        gender_indices = np.random.choice(df_issues.index, size=n_inconsistent, replace=False)
        for idx in gender_indices:
            if pd.notna(df_issues.loc[idx, 'Gender']):
                current = str(df_issues.loc[idx, 'Gender']).strip()
                if current.lower() == 'male':
                    df_issues.loc[idx, 'Gender'] = np.random.choice(['male', 'MALE', 'Male ', ' M', 'M'])
                else:
                    df_issues.loc[idx, 'Gender'] = np.random.choice(['female', 'FEMALE', 'Female ', ' F', 'F'])

    # 4. Data inconsistency in Account_Type
    if 'Account_Type' in df_issues.columns:
        n_inconsistent = int(len(df_issues) * 0.02)
        account_indices = np.random.choice(df_issues.index, size=n_inconsistent, replace=False)
        for idx in account_indices:
            if pd.notna(df_issues.loc[idx, 'Account_Type']):
                current = str(df_issues.loc[idx, 'Account_Type']).strip()
                if current == 'Savings':
                    df_issues.loc[idx, 'Account_Type'] = np.random.choice(['savings', 'SAVINGS', 'Savings ', ' saving'])
                elif current == 'Business':
                    df_issues.loc[idx, 'Account_Type'] = np.random.choice(['business', 'BUSINESS', 'Business ', ' bus'])
                elif current == 'Checking':
                    df_issues.loc[idx, 'Account_Type'] = np.random.choice(['checking', 'CHECKING', 'Checking ', ' check'])

    # 5. Data inconsistency in City
    if 'City' in df_issues.columns:
        n_inconsistent = int(len(df_issues) * 0.02)
        city_indices = np.random.choice(df_issues.index, size=n_inconsistent, replace=False)
        for idx in city_indices:
            if pd.notna(df_issues.loc[idx, 'City']):
                current = str(df_issues.loc[idx, 'City'])
                # Add leading/trailing spaces randomly
                if np.random.random() < 0.5:
                    current = ' ' + current
                else:
                    current = current + ' '
                # Mix case occasionally
                if np.random.random() < 0.3:
                    current = current.upper()
                elif np.random.random() < 0.5:
                    current = current.lower()
                else:
                    current = current.title()
                df_issues.loc[idx, 'City'] = current

    # 6. Add outliers in numerical features
    if 'Age' in df_issues.columns:
        n_outliers = int(len(df_issues) * 0.01)
        age_outlier_indices = np.random.choice(df_issues.index, size=n_outliers, replace=False)
        age_outliers = [150, -5, 0, 999, 120, -1]
        for idx in age_outlier_indices:
            df_issues.loc[idx, 'Age'] = np.random.choice(age_outliers)

    if 'Transaction_Amount' in df_issues.columns:
        n_outliers = int(len(df_issues) * 0.02)
        amount_outlier_indices = np.random.choice(df_issues.index, size=n_outliers, replace=False)
        amount_outliers = [1000000, 5000000, -1000, 9999999, -500]
        for idx in amount_outlier_indices:
            df_issues.loc[idx, 'Transaction_Amount'] = np.random.choice(amount_outliers)

    if 'Account_Balance' in df_issues.columns:
        n_outliers = int(len(df_issues) * 0.02)
        balance_outlier_indices = np.random.choice(df_issues.index, size=n_outliers, replace=False)
        balance_outliers = [10000000, -10000, 99999999, -500000, 0]
        for idx in balance_outlier_indices:
            df_issues.loc[idx, 'Account_Balance'] = np.random.choice(balance_outliers)

    # Reset index to handle new rows
    df_issues = df_issues.reset_index(drop=True)

    return df_issues


def generate_data_quality_report(df_before, df_after):
    """
    Generate a report comparing data quality before and after.

    Parameters
    ----------
    df_before : pandas.DataFrame
        Original clean DataFrame
    df_after : pandas.DataFrame
        DataFrame after introducing issues

    Returns
    -------
    dict
        Dictionary containing statistics about introduced issues
    """
    report = {
        'rows_added': len(df_after) - len(df_before),
        'duplicate_rows': df_after.duplicated().sum(),
        'missing_values_total': df_after.isnull().sum().sum(),
        'missing_by_column': df_after.isnull().sum().to_dict(),
        'inconsistencies_detected': {}
    }

    # Check categorical inconsistencies
    if 'Gender' in df_after.columns:
        unique_genders = set(df_after['Gender'].dropna().astype(str).str.strip().unique())
        report['inconsistencies_detected']['Gender_variations'] = len(unique_genders)

    if 'Account_Type' in df_after.columns:
        unique_accounts = set(df_after['Account_Type'].dropna().astype(str).str.strip().unique())
        report['inconsistencies_detected']['Account_Type_variations'] = len(unique_accounts)

    if 'City' in df_after.columns:
        unique_cities = set(df_after['City'].dropna().astype(str).str.strip().unique())
        report['inconsistencies_detected']['City_variations'] = len(unique_cities)

    # Check outlier counts
    for col in ['Age', 'Transaction_Amount', 'Account_Balance']:
        if col in df_after.columns:
            if col == 'Age':
                outliers = ((df_after[col] < 0) | (df_after[col] > 100)).sum()
            elif col == 'Transaction_Amount':
                outliers = ((df_after[col] < 0) | (df_after[col] > 100000)).sum()
            elif col == 'Account_Balance':
                outliers = ((df_after[col] < 0) | (df_after[col] > 200000)).sum()
            report['inconsistencies_detected'][f'{col}_outliers'] = int(outliers)

    return report


if __name__ == "__main__":
    # Example usage when run directly
    print("Data Issues Generator")
    print("=" * 50)
    print("\nUsage in notebook:")
    print("  from data_issues_generator import introduce_data_issues")
    print("  df_corrupted = introduce_data_issues(df_clean)")
    print("\nTo generate a report:")
    print("  from data_issues_generator import generate_data_quality_report")
    print("  report = generate_data_quality_report(df_clean, df_corrupted)")
