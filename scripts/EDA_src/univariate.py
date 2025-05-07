import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot

def univariate_numeric(df, column, figsize=(10, 6), bins=20):
    """
    Perform univariate analysis for numeric variables
    """
    plt.figure(figsize=figsize)
    
    # Create a subplot grid
    gs = plt.GridSpec(2, 2)
    
    # Histogram
    ax0 = plt.subplot(gs[0, 0])
    sns.histplot(df[column], kde=True, ax=ax0, bins=bins)
    ax0.set_title(f'Distribution of {column}')
    
    # Box plot
    ax1 = plt.subplot(gs[0, 1])
    sns.boxplot(y=df[column], ax=ax1)
    ax1.set_title(f'Boxplot of {column}')
    
    # QQ plot
    ax2 = plt.subplot(gs[1, 0])
    sm.qqplot(df[column].dropna(), line='45', ax=ax2)
    ax2.set_title(f'QQ Plot of {column}')
    
    # Stats summary
    ax3 = plt.subplot(gs[1, 1])
    ax3.axis('off')
    stats_text = f"""
    Statistics for {column}:
    
    Count: {df[column].count()}
    Mean: {df[column].mean():.2f}
    Median: {df[column].median():.2f}
    Std Dev: {df[column].std():.2f}
    Min: {df[column].min():.2f}
    Max: {df[column].max():.2f}
    Skewness: {df[column].skew():.2f}
    Kurtosis: {df[column].kurtosis():.2f}
    Missing: {df[column].isna().sum()} ({df[column].isna().mean()*100:.2f}%)
    """
    ax3.text(0.1, 0.5, stats_text, fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Check for outliers using IQR method
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))]
    
    if not outliers.empty:
        print(f"Potential outliers detected for {column}: {len(outliers)} values")