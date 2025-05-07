import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import glm
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

from .multivariate import correlation_matrix, pca_analysis, cluster_analysis

def analyze_purchase_behavior(df):
    """
    Analyze purchase behavior during crisis
    """
    print("\n" + "="*80)
    print("ANALYZING PURCHASE BEHAVIOR DURING CRISIS")
    print("="*80)
    
    # Check if the purchase indicator exists
    purchase_col = 'opi_purchased?'
    if purchase_col in df.columns:
        # Basic stats
        purchase_rate = df[purchase_col].mean() * 100
        print(f"Online purchase rate during crisis: {purchase_rate:.2f}%")
        
        # Visualize purchase rates
        plt.figure(figsize=(12, 5))
        
        # Purchase rate by gender
        if 'gender_encoded' in df.columns:
            plt.subplot(1, 3, 1)
            purchase_by_gender = df.groupby('gender_encoded')[purchase_col].mean() * 100
            purchase_by_gender.plot(kind='bar')
            plt.title('Purchase Rate by Gender')
            plt.ylabel('Percentage (%)')
            plt.xlabel('Gender (0=Male, 1=Female, 2=Other)')
        
        # Purchase rate by age
        if 'age_encoded' in df.columns:
            plt.subplot(1, 3, 2)
            purchase_by_age = df.groupby('age_encoded')[purchase_col].mean() * 100
            purchase_by_age.plot(kind='bar')
            plt.title('Purchase Rate by Age Group')
            plt.ylabel('Percentage (%)')
            plt.xlabel('Age Group (0=18-25, 1=25-35, 2=35-45, 3=45-55)')
        
        # Purchase satisfaction
        if 'opi_satisfaction' in df.columns:
            plt.subplot(1, 3, 3)
            df.groupby(purchase_col)['opi_satisfaction'].mean().plot(kind='bar')
            plt.title('Avg. Satisfaction by Purchase Status')
            plt.ylabel('Satisfaction Score')
            plt.xlabel('Made Purchase (0=No, 1=Yes)')
        
        plt.tight_layout()
        plt.show()
        
        # Effect of key variables on purchase decision
        print("\nKey factors influencing purchase decision:")
        
        # Create a logistic regression model
        try:
            temp_df = df.copy()
            temp_df.rename(columns={purchase_col: 'purchase_status'}, inplace=True)
            # Select key variables from each construct
            key_vars = []
            for prefix in ['peou_', 'pu_', 'sa_', 'si_', 'att_', 'risk_']:
                avg_col = f'{prefix}avg'
                if avg_col in df.columns:
                    key_vars.append(avg_col)
            
            # Add demographic variables
            demo_vars = ['gender_encoded', 'age_encoded', 'education_encoded']
            key_vars.extend([var for var in demo_vars if var in df.columns])
            
            # Build formula
            formula = f"purchase_status ~ " + " + ".join(key_vars)
            
            # Fit model
            model = glm(formula=formula, data=temp_df, family=sm.families.Binomial()).fit()
            print(model.summary())
        except Exception as e:
            print(f"Could not build logistic regression model: {e}")


def analyze_platform_usage(df):
    """
    Analyze platform usage patterns
    """
    print("\n" + "="*80)
    print("ANALYZING PLATFORM USAGE PATTERNS")
    print("="*80)
    
    # Analyze aggregate usage counts
    usage_cols = ['platform_count', 'pharmacy_count', 'fashion_count', 
                  'grocery_count', 'automobile_count', 'total_services_used']
    
    for col in usage_cols:
        if col in df.columns:
            print(f"\nDistribution of {col}:")
            print(df[col].describe())
    
    # Plot distribution of services used
    plt.figure(figsize=(12, 6))
    
    # Distribution of total services
    if 'total_services_used' in df.columns:
        plt.subplot(1, 2, 1)
        sns.histplot(df['total_services_used'], kde=True)
        plt.title('Distribution of Total Services Used')
        plt.xlabel('Number of Services')
        plt.ylabel('Count')
    
    # Breakdown by service type
    plt.subplot(1, 2, 2)
    service_avgs = []
    labels = []
    
    for col in ['platform_count', 'pharmacy_count', 'fashion_count', 
                'grocery_count', 'automobile_count']:
        if col in df.columns:
            service_avgs.append(df[col].mean())
            labels.append(col.replace('_count', ''))
    
    plt.bar(labels, service_avgs)
    plt.title('Average Usage by Service Type')
    plt.ylabel('Average Number Used')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Top platforms analysis
    platform_cols = [col for col in df.columns if col.startswith('gecp_') and col != 'gecp_None']
    if platform_cols:
        platform_usage = df[platform_cols].sum().sort_values(ascending=False)
        
        plt.figure(figsize=(12, 6))
        platform_usage.head(10).plot(kind='bar')
        plt.title('Top 10 E-Commerce Platforms Used')
        plt.ylabel('Number of Users')
        plt.xlabel('Platform')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def correlation_analysis(df, constructs):
    """
    Perform correlation analysis between key constructs
    """
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS BETWEEN CONSTRUCTS")
    print("="*80)
    
    # Get average scores for each construct
    avg_cols = [col for col in df.columns if col.endswith('avg')]
    
    if len(avg_cols) >= 2:
        # Correlation matrix for construct averages
        correlation_matrix(df, avg_cols, title='Correlation between Constructs')
    
    # Correlation between key constructs and outcomes
    outcome_vars = constructs['opi']
    predictor_vars = []
    
    # Collect one variable from each construct for simplicity
    for construct_name in ['peou', 'pu', 'sa', 'si', 'att', 'risk']:
        if constructs[construct_name]:
            predictor_vars.append(constructs[construct_name][0])
    
    # Create correlation matrix between predictors and outcomes
    all_vars = predictor_vars + outcome_vars
    if len(all_vars) >= 2:
        correlation_matrix(df, all_vars, title='Correlation between Predictors and Outcomes')

def multivariate_analysis(df, constructs):
    """
    Perform advanced multivariate analysis
    """
    print("\n" + "="*80)
    print("ADVANCED MULTIVARIATE ANALYSIS")
    print("="*80)
    
    # PCA on main constructs
    main_vars = []
    for construct_name in ['peou', 'pu', 'sa', 'si', 'att', 'risk']:
        main_vars.extend(constructs[construct_name])
    
    if len(main_vars) >= 3:
        pca_analysis(df, main_vars, n_components=3)
    
    # Cluster analysis on outcome variables
    outcome_vars = constructs['opi']
    if len(outcome_vars) >= 2:
        cluster_analysis(df, outcome_vars)
    
    # Create a visualization of the conceptual model
    plt.figure(figsize=(12, 8))
    plt.title('Conceptual Model Visualization', fontsize=16)
    
    # Draw conceptual model (simplified)
    ax = plt.gca()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Draw boxes for constructs
    constructs_coords = {
        'peou': (2, 8, 'Perceived\nEase of Use'),
        'pu': (2, 6, 'Perceived\nUsefulness'),
        'sa': (2, 4, 'Structural\nAssurance'),
        'si': (2, 2, 'Social\nInfluence'),
        'att': (5, 5, 'Attitude'),
        'risk': (5, 3, 'Perceived\nRisk'),
        'opi': (8, 5, 'Online Purchase\nIntention')
    }
    
    # Draw rectangles
    for construct, (x, y, label) in constructs_coords.items():
        rect = Rectangle((x-1, y-0.5), 2, 1, facecolor='lightblue', alpha=0.8, edgecolor='black')
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontweight='bold')
    
    # Draw arrows
    arrows = [
        (3, 8, 5, 5.2),  # PEOU -> ATT
        (3, 6, 5, 4.8),  # PU -> ATT
        (3, 4, 5, 4.4),  # SA -> ATT
        (3, 2, 5, 3.6),  # SI -> ATT
        (6, 5, 8, 5.2),  # ATT -> OPI
        (6, 3, 8, 4.8)   # RISK -> OPI
    ]
    
    for x1, y1, x2, y2 in arrows:
        ax.arrow(x1, y1, x2-x1-0.2, y2-y1, head_width=0.1, head_length=0.1, fc='black', ec='black', length_includes_head=True)
    
    plt.tight_layout()
    plt.show()