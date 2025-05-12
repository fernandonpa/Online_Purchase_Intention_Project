import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import glm
from matplotlib.patches import Rectangle, FancyBboxPatch
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

def create_conceptual_model(df, constructs):
    """
    Create a visualization of the online purchase intention conceptual model
    with correlation information and styling improvements
    """
    from matplotlib.patches import FancyBboxPatch
    
    # 1. Setup the figure and calculate correlations
    correlations = _calculate_correlations(df)
    fig, ax = _setup_figure()
    
    # 2. Define and draw the constructs (boxes)
    constructs_coords = _get_construct_coordinates()
    _draw_constructs(ax, constructs_coords)
    
    # 3. Define and draw relationship arrows
    arrow_props = _get_arrow_properties()
    _draw_arrows(ax, arrow_props, correlations)
    
    # 4. Add explanatory text and legend
    _add_legend_and_notes(fig)
    
    plt.tight_layout()
    plt.show()
    
    return correlations

def _calculate_correlations(df):
    """Calculate correlations between model constructs"""
    correlations = {}
    construct_avgs = {
        'peou': 'peou_avg', 'pu': 'pu_avg', 'sa': 'sa_avg', 
        'si': 'si_avg', 'att': 'att_avg', 'risk': 'risk_avg'
    }
    
    # Get column names for key measurements
    attitude_col = 'att_avg' if 'att_avg' in df.columns else None
    opi_col = 'opi_satisfaction' if 'opi_satisfaction' in df.columns else None
    risk_col = 'risk_avg' if 'risk_avg' in df.columns else None
    
    # Compute correlations with attitude
    if attitude_col and all(col in df.columns for col in construct_avgs.values()):
        for construct, col in construct_avgs.items():
            if col in df.columns and attitude_col in df.columns:
                correlations[f'{construct}_att'] = df[col].corr(df[attitude_col]).round(2)
    
    # Compute correlations with risk
    if risk_col and all(col in df.columns for col in construct_avgs.values()):
        for construct, col in construct_avgs.items():
            if col in df.columns and risk_col in df.columns and construct != 'risk':
                correlations[f'{construct}_risk'] = df[col].corr(df[risk_col]).round(2)
    
    # Compute correlations with OPI
    if opi_col and attitude_col in df.columns and 'risk_avg' in df.columns:
        correlations['att_opi'] = df[attitude_col].corr(df[opi_col]).round(2)
        correlations['risk_opi'] = df['risk_avg'].corr(df[opi_col]).round(2)
        
    return correlations

def _setup_figure():
    """Create and configure the figure and axes"""
    fig = plt.figure(figsize=(14, 10), facecolor='#f9f9f9')
    plt.title('Conceptual Model Visualization', fontsize=20, fontweight='bold', pad=20)
    
    ax = plt.gca()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_facecolor('#f9f9f9')
    
    return fig, ax

def _get_construct_coordinates():
    """Define the coordinates and labels for each construct"""
    return {
        'peou': (2, 8, 'Perceived\nEase of Use'),
        'pu': (2, 6, 'Perceived\nUsefulness'),
        'sa': (2, 4, 'Structural\nAssurance'),
        'si': (2, 2, 'Social\nInfluence'),
        'att': (5, 5, 'Attitude'),
        'risk': (5, 3, 'Perceived\nRisk'),
        'opi': (8, 5, 'Online Purchase\nIntention')
    }

def _draw_constructs(ax, constructs_coords):
    """Draw the construct boxes with appropriate styling"""
    for construct, (x, y, label) in constructs_coords.items():
        # Determine appearance based on construct type
        if construct in ['att', 'risk']:
            # Mediator variables
            facecolor = '#ADD8E6'  # Light blue
            edgewidth = 2
        elif construct == 'opi':
            # Outcome variable
            facecolor = '#90EE90'  # Light green
            edgewidth = 2
        else:
            # Predictor variables
            facecolor = '#E6E6FA'  # Lavender
            edgewidth = 1.5
        
        # Create and add the box
        rect = FancyBboxPatch(
            (x-1, y-0.5), 2, 1, 
            boxstyle="round,pad=0.3",
            facecolor=facecolor, 
            alpha=0.9, 
            edgecolor='black',
            linewidth=edgewidth,
            zorder=1
        )
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontweight='bold', fontsize=10)

def _get_arrow_properties():
    """Define the properties for relationship arrows"""
    return [
        # Format: (start_x, start_y, end_x, end_y, correlation_key, is_positive)
        (3, 8, 5, 5.2, 'peou_att', True),    # PEOU -> ATT
        (3, 6, 5, 4.8, 'pu_att', True),      # PU -> ATT
        (3, 4, 5, 4.4, 'sa_att', True),      # SA -> ATT
        (3, 2, 5, 3.6, 'si_att', True),      # SI -> ATT
        (3, 2.2, 5, 3.0, 'si_risk', True),   # SI -> RISK
        (6, 5, 8, 5.2, 'att_opi', True),     # ATT -> OPI
        (6, 3, 8, 4.8, 'risk_opi', False),   # RISK -> OPI
        # Additional links
        (3, 8, 5, 3.3, 'peou_risk', False),  # PEOU -> RISK
        (3, 6, 5, 3.2, 'pu_risk', False),    # PU -> RISK
        (3, 4, 5, 3.1, 'sa_risk', False)     # SA -> RISK
    ]

def _draw_arrows(ax, arrow_props, correlations):
    """Draw the relationship arrows with correlation information"""
    for x1, y1, x2, y2, corr_key, is_positive in arrow_props:
        # Get correlation value if available
        corr_value = correlations.get(corr_key, None)
        
        # Determine arrow styling based on correlation
        width, color = _get_arrow_style(corr_value, is_positive)
            
        # Draw the arrow
        ax.annotate(
            '',
            xy=(x2-0.2, y2), 
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle='-|>', 
                lw=width, 
                color=color, 
                shrinkA=5, 
                shrinkB=5,
                connectionstyle='arc3,rad=0.1',
                zorder=2
            )
        )
        
        # Add correlation label if available
        if corr_value is not None:
            _add_correlation_label(ax, x1, y1, x2, y2, corr_value, color)

def _get_arrow_style(corr_value, is_positive):
    """Determine arrow width and color based on correlation"""
    if corr_value is not None:
        # Scale width from 1 to 3 based on correlation strength
        width = 0.5 + abs(corr_value) * 2.5
        # Adjust arrow color based on correlation direction
        if corr_value > 0:
            color = 'darkblue' if is_positive else 'red'
        else:
            color = 'red' if is_positive else 'darkblue'
    else:
        width = 1.5  # Default width
        color = 'black'
        
    return width, color

def _add_correlation_label(ax, x1, y1, x2, y2, corr_value, color):
    """Add a label showing the correlation value"""
    # Calculate midpoint for label position with slight offset
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    offset_x = 0.2
    offset_y = 0.2 if y2 > y1 else -0.2
    
    # Add correlation text
    ax.text(
        mid_x + offset_x, mid_y + offset_y,
        f'r = {corr_value}',
        fontsize=8,
        color=color,
        fontweight='bold',
        ha='center',
        va='center',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2')
    )

def _add_legend_and_notes(fig):
    """Add legend and explanatory notes to the figure"""
    # Add legend explaining the model
    legend_text = (
        'Model Components:\n'
        '• Blue boxes: Mediating variables\n'
        '• Green box: Outcome variable\n'
        '• Arrow width: Correlation strength\n'
        '• Blue arrows: Positive relationships\n'
        '• Red arrows: Negative relationships'
    )
    fig.text(0.02, 0.02, legend_text, fontsize=9, 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Add a research context note
    note_text = (
        'Note: This model illustrates key relationships in online purchase intention.\n'
        'Arrows indicate causal paths, with correlation values (r) where available.'
    )
    fig.text(0.98, 0.02, note_text, fontsize=9, ha='right',
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

# Use in the multivariate_analysis function
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
    
    # Create improved conceptual model visualization
    correlations = create_conceptual_model(df, constructs)
    
    # Print key correlations for interpretation
    if correlations:
        print("\nKey Correlations in Conceptual Model:")
        for relation, value in correlations.items():
            from_var, to_var = relation.split('_')
            print(f"• {from_var.upper()} → {to_var.upper()}: r = {value}")