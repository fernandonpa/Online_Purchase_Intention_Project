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

