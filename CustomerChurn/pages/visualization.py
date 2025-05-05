import streamlit as st
import pandas as pd
import numpy as np
from utils.visualization import (
    plot_data_distribution, 
    plot_churn_rate_by_feature, 
    plot_correlation_matrix,
    plot_customer_segments,
    plot_churn_prediction_probability_distribution
)

def show_visualization():
    """
    Display the visualization page
    """
    st.title("Data Visualization & Insights")
    
    # Check if data is available
    if "data" not in st.session_state or st.session_state.data is None:
        st.warning("No data available. Please upload data first.")
        if st.button("Go to Data Upload"):
            st.session_state.page = "Data Upload"
            st.rerun()
        return
    
    # Get data from session state
    df = st.session_state.data
    # Create tabs for different visualization categories
    tab1, tab2, tab3, tab4 = st.tabs([
        "Feature Distribution", 
        "Churn Analysis", 
        "Feature Relationships",
        "Prediction Analysis"
    ])
    
    # Tab 1: Feature Distribution
    with tab1:
        st.markdown("## Feature Distribution")
        st.markdown("Explore the distribution of different features in your dataset.")
        
        # Select column and chart type
        col1, col2 = st.columns(2)
        
        with col1:
            column = st.selectbox(
                "Select a column:",
                df.columns,
                key="dist_column"
            )
        
        with col2:
            chart_type = st.selectbox(
                "Select chart type:",
                ["histogram", "bar", "pie"],
                key="dist_chart_type"
            )
        
        # Plot distribution
        fig = plot_data_distribution(df, column, chart_type)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Churn Analysis
    with tab2:
        st.markdown("## Churn Analysis")
        
        # Check if target column is selected
        if "target_column" not in st.session_state:
            st.warning("Target column not selected. Please select a target column first.")
            if st.button("Go to Data Upload", key="goto_dataupload_from_tab2"):
                st.session_state.page = "Data Upload"
                st.rerun()
            return
        
        target_column = st.session_state.target_column
        st.markdown(f"Analyze churn rates across different features. Target column: **{target_column}**")
        
        # Select feature for churn analysis
        feature = st.selectbox(
            "Select a feature to analyze churn rate:",
            [col for col in df.columns if col != target_column],
            key="churn_feature"
        )
        # Convert Interval values to string for plotting
        if pd.api.types.is_interval_dtype(df[feature]):
            df[feature] = df[feature].apply(lambda x: f"{x.left} - {x.right}" if pd.notnull(x) else "Unknown"  )


        # Plot churn rate by feature
        fig = plot_churn_rate_by_feature(df, feature, target=target_column)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Churn rate statistics
        st.markdown("### Churn Rate Statistics")
        
        # Convert target to numeric if needed for calculations
        if df[target_column].dtype == 'object':
            if all(val in ['Yes', 'No'] for val in df[target_column].unique()):
                target_numeric = df[target_column].map({'Yes': 1, 'No': 0})
            else:
                try:
                    target_numeric = df[target_column].astype(int)
                except:
                    st.error(f"Could not convert {target_column} column to numeric for calculations.")
                    target_numeric = None
        else:
            target_numeric = df[target_column]
        
        if target_numeric is not None:
            # Overall churn rate
            overall_churn_rate = target_numeric.mean()
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Overall Churn Rate", f"{overall_churn_rate:.2%}")
            
            with col2:
                churn_count = target_numeric.sum()
                st.metric("Total Churned Customers", int(churn_count))
            
            with col3:
                non_churn_count = len(target_numeric) - churn_count
                st.metric("Total Retained Customers", int(non_churn_count))
    
    # Tab 3: Feature Relationships
    with tab3:
        st.markdown("## Feature Relationships")
        st.markdown("Explore relationships between different features in your dataset.")
        
        # Correlation Analysis
        st.markdown("### Correlation Analysis")
        
        corr_method = st.selectbox(
            "Select correlation method:",
            ["pearson", "spearman", "kendall"],
            key="corr_method"
        )
        # Select only numeric columns for correlation
        numeric_df = df.select_dtypes(include=['number'])
        # Optional: Check if there's enough numeric data to proceed
        if numeric_df.shape[1] < 2:
            st.warning("Not enough numeric columns for correlation matrix.")
        else:
            # Plot correlation matrix
            fig = plot_correlation_matrix(numeric_df, method=corr_method)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # Customer Segmentation
        st.markdown("### Customer Segmentation")
        
        # Select features for segmentation
        col1, col2 = st.columns(2)
        
        with col1:
            feature1 = st.selectbox(
                "Select X-axis feature:",
                df.select_dtypes(include=['int64', 'float64']).columns,
                key="segment_feature1"
            )
        
        with col2:
            feature2 = st.selectbox(
                "Select Y-axis feature:",
                [col for col in df.select_dtypes(include=['int64', 'float64']).columns if col != feature1],
                key="segment_feature2"
            )
        
        # Plot customer segments
        if "target_column" in st.session_state:
            fig = plot_customer_segments(df, target=st.session_state.target_column, feature1=feature1, feature2=feature2)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Target column not selected. Please select a target column to see customer segments.")
    
    # Tab 4: Prediction Analysis
    with tab4:
        st.markdown("## Prediction Analysis")
        
        # Check if predictions are available
        if ("predictions" not in st.session_state or st.session_state.predictions is None or
            "probabilities" not in st.session_state or st.session_state.probabilities is None):
            st.warning("No predictions available. Please train a model and generate predictions first.")
            if st.button("Go to Prediction", key="goto_prediction_from_tab4"):
                st.session_state.page = "Prediction"
                st.rerun()
            return
        
        st.markdown("Analyze the predictions made by the model.")
        
        # Probability Distribution
        st.markdown("### Churn Probability Distribution")
        fig = plot_churn_prediction_probability_distribution(st.session_state.probabilities)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # High Risk Customers
        st.markdown("### High Risk Customers")
        
        risk_threshold = st.slider(
            "Risk Threshold (Probability):",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05
        )
        
        # Filter high risk customers
        result_df = df.copy()
        result_df["Churn_Probability"] = st.session_state.probabilities
        high_risk_df = result_df[result_df["Churn_Probability"] >= risk_threshold]
        
        if not high_risk_df.empty:
            st.write(f"Found {len(high_risk_df)} high-risk customers (churn probability ≥ {risk_threshold:.2f})")
            st.dataframe(high_risk_df)
            
            # Download high risk customers
            if st.button("Download High Risk Customer List"):
                csv_data = high_risk_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name="high_risk_customers.csv",
                    mime="text/csv",
                )
        else:
            st.info(f"No customers found with churn probability ≥ {risk_threshold:.2f}")
