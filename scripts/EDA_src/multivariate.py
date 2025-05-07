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


def pca_analysis(df, columns=None, n_components=2, figsize=(16, 7)):
    """
    Perform PCA analysis for dimensionality reduction and visualization
    """
    if columns is None:
        # Use all numeric columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        columns = numeric_cols
    
    # Standardize the data
    data = df[columns].dropna()
    if data.shape[0] < 10:  # Too few observations
        print("Not enough complete observations for PCA analysis")
        return
    
    X = StandardScaler().fit_transform(data)
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X)
    
    # Create DataFrame with principal components
    pca_df = pd.DataFrame(
        data=principal_components,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    # Plot results
    plt.figure(figsize=figsize)
    
    # Plot explained variance
    plt.subplot(1, 2, 1)
    explained_var = pca.explained_variance_ratio_ * 100
    plt.bar(range(n_components), explained_var)
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance (%)')
    plt.title('Explained Variance by Principal Components')
    plt.xticks(range(n_components), [f'PC{i+1}' for i in range(n_components)])
    
    for i, v in enumerate(explained_var):
        plt.text(i, v + 1, f'{v:.1f}%', ha='center')
    
    # If we have at least 2 components, plot the first two
    if n_components >= 2:
        plt.subplot(1, 2, 2)
        plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.7)
        plt.xlabel(f'PC1 ({explained_var[0]:.1f}%)')
        plt.ylabel(f'PC2 ({explained_var[1]:.1f}%)')
        plt.title('First Two Principal Components')
        
        # Add a circle to represent correlation
        circle = plt.Circle((0, 0), 1, fill=False, linestyle='--', color='gray')
        plt.gca().add_patch(circle)
        
        # Plot feature vectors
        features = columns
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        
        for i, feature in enumerate(features):
            if i < len(loadings):  # Safety check
                plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], head_width=0.05, head_length=0.05, fc='red', ec='red')
                plt.text(loadings[i, 0] * 1.15, loadings[i, 1] * 1.15, feature, color='green', ha='center', va='center')
        
        plt.grid(True)
        plt.axis('equal')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary of PCA results
    print("PCA Summary:")
    print(f"Number of components: {n_components}")
    print(f"Total explained variance: {sum(explained_var):.2f}%")
    
    # Display component loadings
    loadings_df = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=columns
    )
    print("\nComponent Loadings:")
    print(loadings_df)
    
    return pca, loadings_df