import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def identify_column_types(df):
    """
    Identify numeric, categorical, and binary columns
    """
    # Exclude timestamp
    data_cols = df.columns.tolist()
    if 'timestamp' in data_cols:
        data_cols.remove('timestamp')
    
    # Identify column types
    numeric_cols = []
    binary_cols = []
    categorical_cols = []
    
    for col in data_cols:
        unique_vals = df[col].nunique()
        if df[col].dtype in ['int64', 'float64']:
            if unique_vals <= 2:
                binary_cols.append(col)
            else:
                numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    
    return {
        'numeric': numeric_cols,
        'binary': binary_cols,
        'categorical': categorical_cols
    }

def create_construct_groups(df):
    """
    Group columns by their prefixes (constructs)
    """
    constructs = {
        'peou': [col for col in df.columns if col.startswith('peou_')],
        'pu': [col for col in df.columns if col.startswith('pu_')],
        'sa': [col for col in df.columns if col.startswith('sa_')],
        'si': [col for col in df.columns if col.startswith('si_')],
        'att': [col for col in df.columns if col.startswith('att_')],
        'risk': [col for col in df.columns if col.startswith('risk_')],
        'opi': [col for col in df.columns if col.startswith('opi_')],
        'platforms': [col for col in df.columns if col.startswith('gecp_')],
        'online_pharmacy': [col for col in df.columns if col.startswith('op_')],
        'fashion_brands': [col for col in df.columns if col.startswith('fabr_')],
        'grocery_delivery': [col for col in df.columns if col.startswith('gds_')],
        'automobile': [col for col in df.columns if col.startswith('sos_automobile_')],
        'demographic': ['gender_encoded', 'age_encoded', 'marital_status_encoded', 
                       'education_encoded', 'used_online_shopping_encoded'] + 
                       [col for col in df.columns if col.startswith('prof_')]
    }
    return constructs

def create_aggregate_features(df):
    """
    Create aggregate features from binary platform/service columns
    """
    # Count number of platforms/services used by each respondent
    df['platform_count'] = df[[col for col in df.columns if col.startswith('gecp_') and col != 'gecp_None']].sum(axis=1)
    df['pharmacy_count'] = df[[col for col in df.columns if col.startswith('op_') and col != 'op_None']].sum(axis=1)
    df['fashion_count'] = df[[col for col in df.columns if col.startswith('fabr_') and col != 'fabr_None']].sum(axis=1)
    df['grocery_count'] = df[[col for col in df.columns if col.startswith('gds_') and col != 'gds_None']].sum(axis=1)
    df['automobile_count'] = df[[col for col in df.columns if col.startswith('sos_automobile_') and col != 'sos_automobile_None']].sum(axis=1)
    
    # Total services used
    df['total_services_used'] = df['platform_count'] + df['pharmacy_count'] + df['fashion_count'] + df['grocery_count'] + df['automobile_count']
    
    # Average scores for each construct
    construct_prefixes = ['peou_', 'pu_', 'sa_', 'si_', 'att_', 'risk_']
    for prefix in construct_prefixes:
        cols = [col for col in df.columns if col.startswith(prefix)]
        if cols:
            df[f'{prefix}avg'] = df[cols].mean(axis=1)
    
    return df

def analyze_construct(df, construct_name, construct_columns, figsize=(15, 8)):
    """
    Analyze a group of related variables (construct)
    """
    print(f"="*80)
    print(f"Analyzing {construct_name.upper()} construct with {len(construct_columns)} variables")
    print(f"="*80)
    
    # Summary statistics
    summary = df[construct_columns].describe().T
    summary['missing'] = df[construct_columns].isna().sum()
    summary['missing_pct'] = (df[construct_columns].isna().sum() / len(df) * 100).round(2)
    print("\nSummary Statistics:")
    print(summary)
    
    # Create distribution plot
    plt.figure(figsize=figsize)
    
    # If there are too many columns, plot only means
    if len(construct_columns) > 10:
        means = df[construct_columns].mean().sort_values()
        plt.figure(figsize=figsize)
        sns.barplot(x=means.index, y=means.values)
        plt.title(f'Mean values for {construct_name} variables')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    else:
        # Create boxplot for all variables in construct
        plt.figure(figsize=figsize)
        df[construct_columns].boxplot()
        plt.title(f'Distribution of {construct_name} variables')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    # Create correlation heatmap
    plt.figure(figsize=figsize)
    corr = df[construct_columns].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", 
                square=True, linewidths=.5)
    plt.title(f'Correlation Matrix for {construct_name} variables')
    plt.tight_layout()
    plt.show()

