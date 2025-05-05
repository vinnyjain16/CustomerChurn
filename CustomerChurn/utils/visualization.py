import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Common color palette for consistent look
COLOR_PALETTE = {
    'primary': '#3498db',
    'secondary': '#2ecc71',
    'accent': '#e74c3c',
    'neutral': '#95a5a6',
    'dark': '#2c3e50',
    'light': '#ecf0f1',
    'highlight': '#f39c12',
    'gradient1': ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6'],
    'gradient2': ['#1abc9c', '#3498db', '#9b59b6', '#e74c3c', '#f1c40f']
}

# Enhanced chart theme
def apply_chart_theme(fig, title=None):
    """Apply consistent theme to charts"""
    fig.update_layout(
        font_family="Helvetica Neue, Arial, sans-serif",
        title={
            'font': {'size': 22, 'color': COLOR_PALETTE['dark'], 'family': "Helvetica Neue, Arial, sans-serif"},
            'x': 0.5,
            'xanchor': 'center',
            'y': 0.95,
            'yanchor': 'top'
        },
        legend={'font': {'size': 12, 'color': COLOR_PALETTE['dark']}},
        margin=dict(l=40, r=40, t=80, b=40),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest',
        xaxis={
            'title': {'font': {'size': 14, 'color': COLOR_PALETTE['dark']}},
            'tickfont': {'size': 12, 'color': COLOR_PALETTE['dark']},
            'gridcolor': COLOR_PALETTE['light'],
            'zerolinecolor': COLOR_PALETTE['neutral']
        },
        yaxis={
            'title': {'font': {'size': 14, 'color': COLOR_PALETTE['dark']}},
            'tickfont': {'size': 12, 'color': COLOR_PALETTE['dark']},
            'gridcolor': COLOR_PALETTE['light'],
            'zerolinecolor': COLOR_PALETTE['neutral']
        }
    )
    
    if title:
        fig.update_layout(title_text=title)
    
    return fig

def plot_data_distribution(df, column, chart_type='histogram'):
    """
    Plot the distribution of a column in the DataFrame with enhanced visuals
    
    Args:
        df: pandas DataFrame
        column: column name
        chart_type: 'histogram', 'pie', or 'bar'
    """
    if df is None or column not in df.columns:
        return None
    
    # Get proper title with capitalization
    title = f"Distribution of {column.replace('_', ' ').title()}"
    
    if chart_type == 'histogram':
        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(df[column]):
            # Enhanced histogram with KDE
            fig = px.histogram(
                df, 
                x=column, 
                marginal="rug",
                opacity=0.7,
                color_discrete_sequence=[COLOR_PALETTE['primary']],
                template="plotly_white"
            )
            
            # Add mean line
            mean_val = df[column].mean()
            fig.add_vline(
                x=mean_val, 
                line_dash="dash", 
                line_color=COLOR_PALETTE['accent'],
                annotation_text=f"Mean: {mean_val:.2f}",
                annotation_position="top right"
            )
        else:
            # For categorical columns, create a count plot
            value_counts = df[column].value_counts()
            fig = px.bar(
                x=value_counts.index, 
                y=value_counts.values,
                color_discrete_sequence=[COLOR_PALETTE['primary']],
                labels={"x": column, "y": "Count"},
                opacity=0.8
            )
            
            # Add data labels
            fig.update_traces(
                texttemplate='%{y}', 
                textposition='outside',
                hovertemplate='%{x}<br>Count: %{y}<extra></extra>'
            )
    
    elif chart_type == 'pie':
        value_counts = df[column].value_counts()
        
        # Create an enhanced pie chart
        fig = px.pie(
            names=value_counts.index, 
            values=value_counts.values,
            hole=0.4,  # Donut chart
            color_discrete_sequence=COLOR_PALETTE['gradient1'],
        )
        
        # Improve pie chart appearance
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hoverinfo='label+percent+value',
            marker=dict(line=dict(color='white', width=2))
        )
    
    elif chart_type == 'bar':
        value_counts = df[column].value_counts().sort_values(ascending=False)
        
        # Enhanced bar chart
        fig = px.bar(
            x=value_counts.index, 
            y=value_counts.values,
            color=value_counts.values,
            color_continuous_scale=px.colors.sequential.Blues,
            labels={"x": column, "y": "Count", "color": "Count"},
            opacity=0.9
        )
        
        # Add data labels
        fig.update_traces(
            texttemplate='%{y}', 
            textposition='outside',
            hovertemplate='%{x}<br>Count: %{y}<extra></extra>'
        )
        
        # Rotate x-axis labels if there are many categories
        if len(value_counts) > 5:
            fig.update_layout(xaxis_tickangle=-45)
    
    else:
        st.error(f"Unsupported chart type: {chart_type}")
        return None
    
    # Apply common theme
    apply_chart_theme(fig, title)
    
    return fig

def plot_churn_rate_by_feature(df, feature, target='Churn'):
    """
    Plot the churn rate by a specific feature with enhanced visuals
    
    Args:
        df: pandas DataFrame
        feature: feature column name
        target: target column name (Churn)
    """
    if df is None or feature not in df.columns or target not in df.columns:
        return None
    
    # Get proper title with capitalization
    title = f"Churn Rate by {feature.replace('_', ' ').title()}"
    
    # Convert target to numeric if needed
    if df[target].dtype == 'object':
        if all(val in ['Yes', 'No'] for val in df[target].unique()):
            target_numeric = df[target].map({'Yes': 1, 'No': 0})
        else:
            try:
                target_numeric = df[target].astype(int)
            except:
                st.error(f"Could not convert {target} column to numeric.")
                return None
    else:
        target_numeric = df[target]
    
    # Create a temporary dataframe with the target
    temp_df = df.copy()
    temp_df['ChurnNumeric'] = target_numeric
    
    # Group by feature and calculate churn rate
    grouped = temp_df.groupby(feature)['ChurnNumeric'].agg(['mean', 'count']).reset_index()
    grouped = grouped.rename(columns={'mean': 'churn_rate', 'count': 'customer_count'})
    
    # Sort by churn rate for better visualization
    grouped = grouped.sort_values('churn_rate', ascending=False)
    
    # Calculate average churn rate for reference
    avg_churn_rate = target_numeric.mean()
    
    # For categorical features
    if pd.api.types.is_object_dtype(df[feature]) or pd.api.types.is_categorical_dtype(df[feature]):
        # Create enhanced bar chart 
        fig = px.bar(
            grouped, 
            x=feature, 
            y='churn_rate',
            color='churn_rate',
            # Removed 'size' parameter that was causing the error
            color_continuous_scale='RdYlGn_r',  # Red for high churn (bad), green for low churn (good)
            text='churn_rate',
            hover_data={
                'customer_count': True,
                'churn_rate': ':.1%'
            },
            labels={
                feature: feature.replace('_', ' ').title(),
                'churn_rate': 'Churn Rate',
                'customer_count': 'Customer Count'
            },
            height=500
        )
        
        # Customize hover template
        fig.update_traces(
            hovertemplate='<b>%{x}</b><br>Churn Rate: %{y:.1%}<br>Customer Count: %{customdata[0]}<extra></extra>',
            texttemplate='%{y:.1%}',
            textposition='outside'
        )
        
        # Add a reference line for average churn rate
        fig.add_hline(
            y=avg_churn_rate, 
            line_dash="dash", 
            line_color="black",
            annotation_text=f"Avg Churn: {avg_churn_rate:.1%}",
            annotation_position="top right"
        )
        
        # Rotate x-axis labels if there are many categories
        if len(grouped) > 5:
            fig.update_layout(xaxis_tickangle=-45)
    
    # For numeric features
    else:
        # Create bins for numeric features
        num_bins = min(10, len(grouped))
        bin_labels = []
        
        # Check if feature is integer with few unique values
        if pd.api.types.is_integer_dtype(df[feature]) and df[feature].nunique() <= 10:
            # Use original values for integers with few unique values
            binned = pd.cut(
                df[feature], 
                bins=sorted(df[feature].unique()) + [df[feature].max() + 1],
                include_lowest=True,
                labels=[str(x) for x in sorted(df[feature].unique())]
            )
            bin_labels = [str(x) for x in sorted(df[feature].unique())]
        else:
            # Create equal-width bins for general numeric data
            bin_edges = np.linspace(df[feature].min(), df[feature].max(), num_bins + 1)
            binned = pd.cut(df[feature], bins=bin_edges, include_lowest=True)
            bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(bin_edges)-1)]
        
        # Add binned column to the dataframe
        temp_df['binned'] = binned
        
        # Group by bins
        bin_groups = temp_df.groupby('binned')['ChurnNumeric'].agg(['mean', 'count']).reset_index()
        bin_groups = bin_groups.rename(columns={'mean': 'churn_rate', 'count': 'customer_count'})
        
        # Create custom bar chart for numeric feature
        fig = px.bar(
            bin_groups, 
            x='binned', 
            y='churn_rate',
            color='churn_rate',
            color_continuous_scale='RdYlGn_r',
            text='churn_rate',
            hover_data={'customer_count': True},
            labels={
                'binned': feature.replace('_', ' ').title(),
                'churn_rate': 'Churn Rate',
                'customer_count': 'Customer Count'
            },
            height=500
        )
        
        # Add line trace showing trend
        fig.add_trace(
            go.Scatter(
                x=bin_groups['binned'],
                y=bin_groups['churn_rate'],
                mode='lines+markers',
                line=dict(color='rgba(0,0,0,0.5)', width=2),
                marker=dict(size=8, color='rgba(0,0,0,0.8)'),
                name='Trend'
            )
        )
        
        # Improve text display
        fig.update_traces(
            texttemplate='%{y:.1%}',
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Churn Rate: %{y:.1%}<br>Customer Count: %{customdata[0]}<extra></extra>',
            selector=dict(type='bar')
        )
        
        # Add a reference line for average churn rate
        fig.add_hline(
            y=avg_churn_rate, 
            line_dash="dash", 
            line_color="black",
            annotation_text=f"Avg Churn: {avg_churn_rate:.1%}",
            annotation_position="top right"
        )
        
        # Rotate x-axis labels
        fig.update_layout(xaxis_tickangle=-45)
    
    # Apply common theme
    apply_chart_theme(fig, title)
    
    # Update y-axis to show percentages
    fig.update_layout(yaxis=dict(tickformat='.0%'))
    
    return fig

def plot_correlation_matrix(df, method='pearson'):
    """
    Plot an enhanced correlation matrix for numeric columns
    
    Args:
        df: pandas DataFrame
        method: correlation method ('pearson', 'spearman', 'kendall')
    """
    if df is None:
        return None
    
    # Get proper title
    title = f"{method.capitalize()} Correlation Matrix"
    
    # Select numeric columns
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    
    if numeric_df.empty:
        st.warning("No numeric columns found for correlation analysis.")
        return None
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr(method=method)
    
    # Create an enhanced heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        aspect="auto",
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1
    )
    
    # Enhance text formatting
    fig.update_traces(
        texttemplate='%{z:.2f}',
        hovertemplate='%{y} & %{x}<br>Correlation: %{z:.3f}<extra></extra>'
    )
    
    # Format column names
    formatted_cols = [col.replace('_', ' ').title() for col in corr_matrix.columns]
    
    # Update axis labels
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(formatted_cols))),
            ticktext=formatted_cols,
            tickangle=-45
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(formatted_cols))),
            ticktext=formatted_cols
        )
    )
    
    # Add a color scale indicator
    fig.update_layout(
        coloraxis_colorbar=dict(
            title=dict(text = 'Correlation', side = 'right'),
           
            ticks='outside',
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=['Perfect<br>Negative<br>(-1)', 'Strong<br>Negative', 'No<br>Correlation<br>(0)', 
                     'Strong<br>Positive', 'Perfect<br>Positive<br>(1)'],
            lenmode='fraction', len=0.8
        )
    )
    
    # Add diagonal line effect
    for i in range(len(corr_matrix)):
        fig.add_shape(
            type="rect",
            x0=i-0.5, y0=i-0.5, x1=i+0.5, y1=i+0.5,
            line=dict(color="black", width=1.5),
            fillcolor="rgba(0,0,0,0)",
        )
    
    # Apply common theme
    apply_chart_theme(fig, title)
    
    # Add custom height for better visualization
    if len(corr_matrix) > 8:
        fig.update_layout(height=700)
    else:
        fig.update_layout(height=600)
    
    return fig

def plot_feature_importance(feature_importance_df, top_n=10):
    """
    Plot feature importance
    
    Args:
        feature_importance_df: DataFrame with feature importance
        top_n: number of top features to show
    """
    if feature_importance_df is None:
        return None
    
    # Get top N features
    top_features = feature_importance_df.head(top_n).copy()
    
    # Sort by importance
    top_features = top_features.sort_values('importance')
    
    # Create horizontal bar chart
    fig = px.bar(
        top_features,
        y='feature',
        x='importance',
        orientation='h',
        title=f"Top {top_n} Features by Importance",
        color='importance',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        yaxis_title="",
        xaxis_title="Importance",
        height=400
    )
    
    return fig

def plot_confusion_matrix(conf_matrix, labels=None):
    """
    Plot confusion matrix
    
    Args:
        conf_matrix: confusion matrix (2D array)
        labels: class labels
    """
    if conf_matrix is None:
        return None
    
    if labels is None:
        labels = ['No Churn', 'Churn']
    
    # Create figure
    fig = px.imshow(
        conf_matrix,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=labels,
        y=labels,
        color_continuous_scale='Blues',
        title="Confusion Matrix"
    )
    
    return fig

def plot_roc_curve(y_true, y_prob):
    """
    Plot ROC curve
    
    Args:
        y_true: true labels
        y_prob: predicted probabilities
    """
    if y_true is None or y_prob is None:
        return None
    
    # Check if the input is empty
    if len(y_true) == 0 or len(y_prob) == 0:
        st.warning("No data available for ROC curve")
        return None
    
    # Handle different input types (Series, arrays, lists)
    if hasattr(y_true, 'values'):  # If it's a pandas Series
        y_true_values = y_true.values
    else:
        y_true_values = np.array(y_true)
    
    # Handle string labels by converting them to binary
    if y_true_values.dtype == np.dtype('O'):  # If object type (strings)
        # Convert 'Yes'/'No' to 1/0
        y_true_binary = np.array([1 if str(label).lower() in ['yes', 'true', '1', 'y'] else 0 for label in y_true_values])
    else:
        y_true_binary = y_true_values
    
    # Calculate ROC curve with explicit positive label
    fpr, tpr, _ = roc_curve(y_true_binary, y_prob, pos_label=1)
    roc_auc = auc(fpr, tpr)
    
    # Create figure
    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC = {roc_auc:.3f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=700, height=500
    )
    
    # Add diagonal line
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    
    return fig

def plot_customer_segments(df, target='Churn', feature1='MonthlyCharges', feature2='TotalCharges'):
    """
    Plot customer segments based on two features
    
    Args:
        df: pandas DataFrame
        target: target column name
        feature1: first feature for segmentation
        feature2: second feature for segmentation
    """
    if df is None or feature1 not in df.columns or feature2 not in df.columns or target not in df.columns:
        return None
    
    # Create scatter plot
    fig = px.scatter(
        df,
        x=feature1,
        y=feature2,
        color=target,
        title=f"Customer Segmentation by {feature1} and {feature2}",
        color_discrete_sequence=["green", "red"],
        opacity=0.7,
        size_max=10
    )
    
    return fig

def plot_model_metrics_comparison(metrics_list, model_names):
    """
    Plot comparison of model metrics
    
    Args:
        metrics_list: list of metrics dictionaries for different models
        model_names: list of model names
    """
    if not metrics_list or not model_names:
        return None
    
    # Extract common metrics
    metric_names = ['accuracy', 'precision', 'recall', 'f1']
    data = {}
    
    for metric in metric_names:
        data[metric] = [m.get(metric, 0) for m in metrics_list]
    
    # Create bar chart
    fig = go.Figure()
    
    for metric, values in data.items():
        fig.add_trace(go.Bar(
            x=model_names,
            y=values,
            name=metric.capitalize()
        ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Model",
        yaxis_title="Score",
        barmode='group',
        yaxis=dict(range=[0, 1]),
        legend_title="Metric"
    )
    
    return fig

def plot_churn_prediction_probability_distribution(probabilities):
    """
    Plot the distribution of churn prediction probabilities
    
    Args:
        probabilities: array of predicted probabilities
    """
    if probabilities is None:
        return None
    
    # Create histogram
    fig = px.histogram(
        x=probabilities,
        nbins=20,
        labels={'x': 'Churn Probability'},
        title="Distribution of Churn Probabilities",
        color_discrete_sequence=['blue']
    )
    
    fig.update_layout(
        xaxis_title="Churn Probability",
        yaxis_title="Count",
        xaxis=dict(range=[0, 1])
    )
    
    return fig
