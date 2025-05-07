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

