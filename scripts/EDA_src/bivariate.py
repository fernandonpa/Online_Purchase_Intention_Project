import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def bivariate_numeric_numeric(df, x_col, y_col, figsize=(10, 6)):
    """
    Perform bivariate analysis for two numeric variables
    """
    plt.figure(figsize=figsize)
    
    # Create a subplot grid
    gs = plt.GridSpec(2, 2)
    
    # Scatter plot
    ax0 = plt.subplot(gs[0, 0])
    sns.scatterplot(x=x_col, y=y_col, data=df, ax=ax0)
    ax0.set_title(f'Scatter plot: {x_col} vs {y_col}')
    
    # Add regression line
    ax1 = plt.subplot(gs[0, 1])
    sns.regplot(x=x_col, y=y_col, data=df, ax=ax1)
    ax1.set_title(f'Regression line: {x_col} vs {y_col}')
    
    # Hexbin plot for dense data
    ax2 = plt.subplot(gs[1, 0])
    hb = ax2.hexbin(df[x_col], df[y_col], gridsize=15, cmap='Blues')
    plt.colorbar(hb, ax=ax2)
    ax2.set_title(f'Hexbin plot: {x_col} vs {y_col}')
    
    # Stats summary
    ax3 = plt.subplot(gs[1, 1])
    ax3.axis('off')
    
    # Calculate correlation coefficients
    pearson_corr, pearson_p = stats.pearsonr(df[x_col].dropna(), df[y_col].dropna())
    spearman_corr, spearman_p = stats.spearmanr(df[x_col].dropna(), df[y_col].dropna())
    
    stats_text = f"""
    Correlation Statistics:
    
    Pearson: {pearson_corr:.3f} (p-value: {pearson_p:.4f})
    Spearman: {spearman_corr:.3f} (p-value: {spearman_p:.4f})
    
    Interpretation:
    - Perfect: ±1.0
    - Strong: ±0.7 to ±1.0
    - Moderate: ±0.4 to ±0.7
    - Weak: ±0.1 to ±0.4
    - None: 0.0 to ±0.1
    """
    ax3.text(0.1, 0.5, stats_text, fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary of the relationship
    print(f"\nRelationship between {x_col} and {y_col}:")
    if abs(pearson_corr) > 0.7:
        strength = "strong"
    elif abs(pearson_corr) > 0.4:
        strength = "moderate"
    elif abs(pearson_corr) > 0.1:
        strength = "weak"
    else:
        strength = "very weak or no"
    
    direction = "positive" if pearson_corr > 0 else "negative"
    significance = "statistically significant" if pearson_p < 0.05 else "not statistically significant"
    
    print(f"There is a {strength} {direction} correlation ({pearson_corr:.3f}) that is {significance} (p={pearson_p:.4f}).")

def bivariate_categorical_numeric(df, cat_col, num_col, figsize=(12, 6)):
    """
    Perform bivariate analysis for categorical and numeric variables
    """
    plt.figure(figsize=figsize)
    
    # Check if categorical variable has too many categories
    n_categories = df[cat_col].nunique()
    if n_categories > 10:
        print(f"Warning: {cat_col} has {n_categories} categories. Showing only top 10 by frequency.")
        top_cats = df[cat_col].value_counts().nlargest(10).index
        df_subset = df[df[cat_col].isin(top_cats)].copy()
    else:
        df_subset = df.copy()
    
    # Create a subplot grid
    gs = plt.GridSpec(1, 2)
    
    # Box plot
    ax0 = plt.subplot(gs[0, 0])
    sns.boxplot(x=cat_col, y=num_col, data=df_subset, ax=ax0)
    ax0.set_title(f'Boxplot: {num_col} by {cat_col}')
    plt.xticks(rotation=45, ha='right')
    
    # Violin plot
    ax1 = plt.subplot(gs[0, 1])
    sns.violinplot(x=cat_col, y=num_col, data=df_subset, ax=ax1)
    ax1.set_title(f'Violin plot: {num_col} by {cat_col}')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate and display statistics
    print("\nGroup Statistics:")
    group_stats = df_subset.groupby(cat_col)[num_col].agg(['count', 'mean', 'std', 'min', 'median', 'max'])
    print(group_stats)
    
    # Perform ANOVA to check if groups are significantly different
    categories = df_subset[cat_col].unique()
    
    if len(categories) > 1:  # ANOVA requires at least 2 groups
        samples = [df_subset[df_subset[cat_col] == cat][num_col].dropna() for cat in categories]
        f_stat, p_value = stats.f_oneway(*samples)
        
        print(f"\nANOVA Test Results:")
        print(f"F-statistic: {f_stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print(f"There is a statistically significant difference in {num_col} across {cat_col} groups (p < 0.05).")
        else:
            print(f"There is no statistically significant difference in {num_col} across {cat_col} groups (p >= 0.05).")

def bivariate_categorical_categorical(df, cat_col1, cat_col2, figsize=(12, 8)):
    """
    Perform bivariate analysis for two categorical variables
    """
    # Check if either variable has too many categories
    n_cat1 = df[cat_col1].nunique()
    n_cat2 = df[cat_col2].nunique()
    
    if n_cat1 > 10 or n_cat2 > 10:
        print(f"Warning: {cat_col1} has {n_cat1} categories and {cat_col2} has {n_cat2} categories.")
        print("Limiting to top categories by frequency.")
        
        if n_cat1 > 10:
            top_cat1 = df[cat_col1].value_counts().nlargest(10).index
            df_subset = df[df[cat_col1].isin(top_cat1)].copy()
        else:
            df_subset = df.copy()
            
        if n_cat2 > 10:
            top_cat2 = df_subset[cat_col2].value_counts().nlargest(10).index
            df_subset = df_subset[df_subset[cat_col2].isin(top_cat2)].copy()
    else:
        df_subset = df.copy()
    
    # Create contingency table
    contingency_table = pd.crosstab(
        df_subset[cat_col1], 
        df_subset[cat_col2],
        normalize='all'
    ) * 100  # Convert to percentages
    
    plt.figure(figsize=figsize)
    
    # Heatmap of the contingency table
    sns.heatmap(contingency_table, annot=True, fmt='.1f', cmap='YlGnBu', linewidths=.5)
    plt.title(f'Heatmap of {cat_col1} vs {cat_col2} (% of Total)')
    plt.tight_layout()
    plt.show()
    
    # Chi-square test for independence
    contingency = pd.crosstab(df_subset[cat_col1], df_subset[cat_col2])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    
    print("\nContingency Table (Counts):")
    print(contingency)
    
    print("\nChi-square Test for Independence:")
    print(f"Chi-square value: {chi2:.4f}")
    print(f"p-value: {p:.4f}")
    print(f"Degrees of freedom: {dof}")
    
    if p < 0.05:
        print(f"There is a statistically significant association between {cat_col1} and {cat_col2} (p < 0.05).")
    else:
        print(f"There is no statistically significant association between {cat_col1} and {cat_col2} (p >= 0.05).")
