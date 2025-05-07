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

def univariate_categorical(df, column, figsize=(10, 6), max_categories=20):
    """
    Perform univariate analysis for categorical variables
    """
    value_counts = df[column].value_counts()
    
    plt.figure(figsize=figsize)
    
    # If too many categories, show only the top ones
    if len(value_counts) > max_categories:
        top_categories = value_counts.nlargest(max_categories)
        other_count = value_counts.sum() - top_categories.sum()
        
        # Add "Other" category
        top_categories = pd.concat([top_categories, pd.Series([other_count], index=["Other"])])
        
        print(f"Showing top {max_categories} categories for {column} (out of {len(value_counts)} total)")
        
        # Create bar plot
        ax = sns.barplot(x=top_categories.index, y=top_categories.values)
        plt.title(f'Distribution of {column} (Top {max_categories} categories)')
    else:
        # Create bar plot for all categories
        ax = sns.barplot(x=value_counts.index, y=value_counts.values)
        plt.title(f'Distribution of {column}')
    
    # Rotate x-labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Categories')
    plt.ylabel('Count')
    
    # Add percentage labels on top of bars
    total = len(df)
    for i, p in enumerate(ax.patches):
        percentage = 100 * p.get_height() / total
        ax.annotate(f'{percentage:.1f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha = 'center', va = 'bottom', 
                    xytext = (0, 5), textcoords = 'offset points')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\nStatistics for {column}:")
    print(f"Total count: {df[column].count()}")
    print(f"Number of unique values: {df[column].nunique()}")
    print(f"Missing values: {df[column].isna().sum()} ({df[column].isna().mean()*100:.2f}%)")
    
    # Print frequency table
    if len(value_counts) <= 20:
        freq_df = pd.DataFrame({
            'Count': value_counts,
            'Percentage': (value_counts / total * 100).round(2)
        })
        print("\nFrequency Table:")
        print(freq_df)


def univariate_binary(df, column, figsize=(10, 5)):
    """
    Perform univariate analysis for binary variables
    """
    plt.figure(figsize=figsize)
    
    # Create pie chart
    ax1 = plt.subplot(1, 2, 1)
    value_counts = df[column].value_counts()
    ax1.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%',
            startangle=90, explode=[0.05] * len(value_counts))
    ax1.set_title(f'Distribution of {column}')
    
    # Create bar chart
    ax2 = plt.subplot(1, 2, 2)
    sns.countplot(x=column, data=df, ax=ax2)
    ax2.set_title(f'Count of {column}')
    
    # Add counts on top of bars
    for p in ax2.patches:
        ax2.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha = 'center', va = 'bottom', 
                    xytext = (0, 5), textcoords = 'offset points')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    total = len(df)
    print(f"\nStatistics for {column}:")
    print(f"Total count: {df[column].count()}")
    print(f"Missing values: {df[column].isna().sum()} ({df[column].isna().mean()*100:.2f}%)")
    
    # Print frequency table
    freq_df = pd.DataFrame({
        'Value': value_counts.index,
        'Count': value_counts.values,
        'Percentage': (value_counts.values / total * 100).round(2)
    })
    print("\nFrequency Table:")
    print(freq_df)
