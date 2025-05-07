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