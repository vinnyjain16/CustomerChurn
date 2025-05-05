import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processor import preprocess_data, get_feature_names_from_preprocessor, export_results
from utils.ml_models import train_model, predict_churn, get_feature_importance
from utils.visualization import plot_confusion_matrix, plot_roc_curve, plot_model_metrics_comparison
from utils.database import save_prediction_to_db, get_recent_predictions, get_model_performance_history

def show_prediction():
    """
    Display the prediction page
    """
    st.title("Customer Churn Prediction")
    
    # Check if data is available
    if "data" not in st.session_state or st.session_state.data is None:
        st.warning("No data available. Please upload data or load sample data first.")
        if st.button("Go to Data Upload"):
            st.session_state.page = "Data Upload"
            st.rerun()
        return
    
    # Check if target column is selected
    if "target_column" not in st.session_state:
        st.warning("Target column not selected. Please select a target column first.")
        if st.button("Go to Data Upload"):
            st.session_state.page = "Data Upload"
            st.rerun()
        return
    
    # Get data and target column
    df = st.session_state.data
    target_column = st.session_state.target_column
    
    # Model training section
    st.markdown("## Model Training")
    
    # Model selection
    st.markdown("### Select Model")
    model_type = st.selectbox(
        "Choose a model type:",
        ["random_forest", "gradient_boost", "logistic", "xgboost"],
        format_func=lambda x: {
            "random_forest": "Random Forest",
            "gradient_boost": "Gradient Boosting",
            "logistic": "Logistic Regression",
            "xgboost": "XGBoost"
        }[x]
    )
    
    # Train test split ratio
    test_size = st.slider("Test set size (%):", 10, 50, 20) / 100
    
    # Train model button
    if st.button("Train Model"):
        with st.spinner("Preprocessing data..."):
            # Preprocess data
            X, y, preprocessor = preprocess_data(df, target_column)
            
            if X is not None and y is not None:
                # Store preprocessor in session state
                st.session_state.preprocessor = preprocessor
                
                # Get feature names
                feature_names = get_feature_names_from_preprocessor(preprocessor, df.drop(columns=[target_column]))
                st.session_state.feature_names = feature_names
                
                # Train model
                model, X_train, X_test, y_train, y_test, metrics = train_model(
                    X, y, model_type=model_type, test_size=test_size
                )
                
                # Store model and metrics in session state
                st.session_state.model = model
                st.session_state.metrics = metrics
                st.session_state.test_data = (X_test, y_test)
                
                st.success("Model trained successfully!")
    
    # Model evaluation
    if "model" in st.session_state and st.session_state.model is not None:
        st.markdown("## Model Evaluation")
        
        metrics = st.session_state.metrics
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
        
        with col2:
            st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
        
        with col3:
            st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
        
        with col4:
            st.metric("F1 Score", f"{metrics.get('f1', 0):.3f}")
        
        # Cross-validation results
        st.markdown("### Cross-Validation")
        st.info(f"Mean CV Accuracy: {metrics.get('cv_mean', 0):.3f} (±{metrics.get('cv_std', 0):.3f})")
        
        # Confusion Matrix
        st.markdown("### Confusion Matrix")
        X_test, y_test = st.session_state.test_data
        conf_matrix = metrics.get('confusion_matrix')
        if conf_matrix is not None:
            cm_fig = plot_confusion_matrix(conf_matrix)
            st.plotly_chart(cm_fig, use_container_width=True)
        else:
            st.warning("Confusion matrix not available.")
        
        # ROC Curve (if applicable)
        if metrics is not None and metrics.get('roc_auc') is not None:
            st.markdown("### ROC Curve")
            y_prob = st.session_state.model.predict_proba(X_test)[:, 1]
            roc_fig = plot_roc_curve(y_test, y_prob)
            st.plotly_chart(roc_fig, use_container_width=True)
        
        # Store multiple models for comparison
        if "model_metrics" not in st.session_state:
            st.session_state.model_metrics = {}
        
        # Update metrics for current model
        model_name = {
            "random_forest": "Random Forest",
            "gradient_boost": "Gradient Boosting",
            "logistic": "Logistic Regression",
            "xgboost": "XGBoost"
        }[model_type]
        
        st.session_state.model_metrics[model_name] = metrics
        
        # Show model comparison if more than one model
        if len(st.session_state.model_metrics) > 1:
            st.markdown("### Model Comparison")
            model_names = list(st.session_state.model_metrics.keys())
            metrics_list = [st.session_state.model_metrics[name] for name in model_names]
            
            comparison_fig = plot_model_metrics_comparison(metrics_list, model_names)
            st.plotly_chart(comparison_fig, use_container_width=True)
    
    # Prediction on full dataset
    st.markdown("## Predict on Full Dataset")
    
    if "model" in st.session_state and st.session_state.model is not None:
        if st.button("Generate Predictions"):
            with st.spinner("Generating predictions..."):
                # Preprocess the data
                X, _, _ = preprocess_data(df, target_column)
                
                # Make predictions
                predictions, probabilities = predict_churn(st.session_state.model, X)
                
                # Store predictions in session state
                st.session_state.predictions = predictions
                st.session_state.probabilities = probabilities
                
                st.success("Predictions generated successfully!")
        
        # Show predictions if available
        if "predictions" in st.session_state and st.session_state.predictions is not None:
            st.markdown("### Prediction Results")
            
            # Summary of predictions
            churn_count = np.sum(st.session_state.predictions)
            non_churn_count = len(st.session_state.predictions) - churn_count
            churn_rate = churn_count / len(st.session_state.predictions)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Predicted Churn Rate", f"{churn_rate:.2%}")
            
            with col2:
                st.metric("Churn Count", int(churn_count))
            
            with col3:
                st.metric("Non-Churn Count", int(non_churn_count))
            
            # Show sample of predictions
            result_df = df.copy()
            result_df["Predicted_Churn"] = st.session_state.predictions
            
            if "probabilities" in st.session_state and st.session_state.probabilities is not None:
                result_df["Churn_Probability"] = st.session_state.probabilities
            
            st.markdown("### Sample Predictions")
            st.dataframe(result_df.head(10))
            
            # Download predictions
            st.markdown("### Download Predictions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Download as CSV"):
                    csv_data = export_results(df, st.session_state.predictions, format='csv')
                    st.download_button(
                        label="Download CSV File",
                        data=csv_data,
                        file_name="churn_predictions.csv",
                        mime="text/csv",
                    )
            
            with col2:
                if st.button("Explore Visualizations"):
                    st.session_state.page = "Visualization"
                    st.rerun()
    else:
        st.info("Please train a model first to generate predictions.")
