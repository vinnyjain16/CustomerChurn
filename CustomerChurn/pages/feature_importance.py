import streamlit as st
import pandas as pd
from utils.ml_models import get_feature_importance
from utils.visualization import plot_feature_importance

def show_feature_importance():
    """
    Display the feature importance page
    """
    st.title("Feature Importance Analysis")
    
    # Check if model is available
    if "model" not in st.session_state or st.session_state.model is None:
        st.warning("No model available. Please train a model first.")
        if st.button("Go to Prediction"):
            st.session_state.page = "Prediction"
            st.rerun()
        return
    
    # Check if feature names are available
    if "feature_names" not in st.session_state:
        st.warning("Feature names not available. Please train a model first.")
        if st.button("Go to Prediction", key="goto_prediction_btn"):
            st.session_state.page = "Prediction"
            st.rerun()
        return
    
    # Get model and feature names
    model = st.session_state.model
    feature_names = st.session_state.feature_names
    
    # Get feature importance
    feature_importance = get_feature_importance(model, feature_names)
    
    if feature_importance is None:
        st.error("Could not extract feature importance from the model.")
        return
    
    # Display feature importance overview
    st.markdown("## Feature Importance Overview")
    st.markdown(
        "Feature importance indicates how influential each feature is in predicting customer churn. "
        "Higher values suggest the feature has a stronger influence on the prediction."
    )
    
    # Number of top features to display
    top_n = st.slider("Number of top features to display:", 5, 30, 10)
    
    # Plot feature importance
    fig = plot_feature_importance(feature_importance, top_n=top_n)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance table
    st.markdown("### Feature Importance Table")
    
    # Display full table with search and sort capabilities
    st.dataframe(
        feature_importance.rename(columns={'importance': 'Importance', 'feature': 'Feature'})
        .reset_index(drop=True),
        hide_index=True
    )
    
    # Key insights
    st.markdown("## Key Insights")
    
    # Get top 5 features
    top_features = feature_importance.head(5)['feature'].tolist()
    
    st.markdown("### Top Influencing Factors")
    st.markdown(
        "The following factors have the strongest influence on customer churn predictions. "
        "Focus on these areas to maximize customer retention efforts."
    )
    
    for i, feature in enumerate(top_features):
        st.markdown(f"**{i+1}. {feature}**")
    
    # Recommendations based on top features
    st.markdown("### Recommendations")
    st.markdown(
        "Based on the feature importance analysis, consider the following actions to reduce customer churn:"
    )
    
    recommendations = [
        "Focus on improving customer experience in areas related to the top factors",
        "Develop targeted retention strategies for high-risk segments",
        "Review pricing and contract terms if they appear as important factors",
        "Enhance service quality in areas that strongly influence churn",
        "Regularly monitor changes in feature importance to adapt strategies"
    ]
    
    for rec in recommendations:
        st.markdown(f"- {rec}")
    
    # Download feature importance data
    st.markdown("### Export Feature Importance Data")
    
    if st.button("Download Feature Importance Data"):
        csv_data = feature_importance.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="feature_importance.csv",
            mime="text/csv",
        )
