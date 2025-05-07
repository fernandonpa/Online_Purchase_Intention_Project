import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def correlation_matrix(df, columns=None, figsize=(12, 10), title='Correlation Matrix'):
    """
    Plot correlation matrix for selected variables
    """
    if columns is None:
        # Use all numeric columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        columns = numeric_cols
    
    corr = df[columns].corr()
    
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Create heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                square=True, linewidths=.5)
    
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
    # Find strong correlations
    strong_corr = corr.unstack()
    strong_corr = strong_corr[strong_corr < 1.0]  # Remove self-correlations
    strong_corr = strong_corr[abs(strong_corr) > 0.5]  # Filter for strong correlations
    
    if not strong_corr.empty:
        strong_corr = strong_corr.sort_values(ascending=False)
        print("Strong correlations (|r| > 0.5):")
        print(strong_corr)
    else:
        print("No strong correlations (|r| > 0.5) found.")