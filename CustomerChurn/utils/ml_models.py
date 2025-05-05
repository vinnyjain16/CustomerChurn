import numpy as np
import pandas as pd
import streamlit as st
import joblib
import datetime
import uuid

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import XGBClassifier

from utils.database import save_model_metadata

# Map models to their parameter dictionaries
MODEL_PARAMS = {
    "logistic": {
        "penalty": "l2",
        "C": 1.0,
        "max_iter": 1000,
        "solver": "lbfgs"
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "bootstrap": True
    },
    "gradient_boost": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "subsample": 1.0
    },
    "xgboost": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "gamma": 0
    }
}

def train_model(X, y, model_type="random_forest", test_size=0.2, random_state=42, save_to_db=True):
    """
    Train a machine learning model for churn prediction
    
    Args:
        X: preprocessed features
        y: target variable
        model_type: type of model to train ('logistic', 'random_forest', 'gradient_boost', 'xgboost')
        test_size: proportion of data to use for testing
        random_state: random seed for reproducibility
        save_to_db: whether to save model metadata to database
        
    Returns:
        model: the trained model
        X_train, X_test, y_train, y_test: data splits
        metrics: dictionary of evaluation metrics
    """
    if X is None or y is None:
        return None, None, None, None, None, None
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    # Initialize the selected model
    if model_type == "logistic":
        model = LogisticRegression(random_state=random_state, max_iter=1000)
    elif model_type == "random_forest":
        model = RandomForestClassifier(random_state=random_state, n_estimators=100)
    elif model_type == "gradient_boost":
        model = GradientBoostingClassifier(random_state=random_state, n_estimators=100)
    elif model_type == "xgboost":
        model = XGBClassifier(random_state=random_state, n_estimators=100)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Train the model with progress bar
    with st.spinner(f"Training {model_type.replace('_', ' ').title()} model..."):
        model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Evaluate the model
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0, average='macro'),
        'recall': recall_score(y_test, y_pred, zero_division=0, average='macro'),
        'f1': f1_score(y_test, y_pred, zero_division=0, average='macro'),
    }
    
    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)
    
    # Classification report
    metrics['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    metrics['cv_mean'] = cv_scores.mean()
    metrics['cv_std'] = cv_scores.std()
    
    # Generate model version
    model_version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model metadata to database if requested
    if save_to_db:
        params = MODEL_PARAMS.get(model_type, {})
        success, model_id = save_model_metadata(
            model_type=model_type,
            model_version=model_version,
            metrics=metrics,
            model_params=params
        )
        
        if success:
            st.success(f"Model metadata saved to database (ID: {model_id})")
        else:
            st.warning(f"Failed to save model metadata: {model_id}")
    
    return model, X_train, X_test, y_train, y_test, metrics

def predict_churn(model, X):
    """
    Make churn predictions using the trained model
    
    Args:
        model: trained model
        X: preprocessed features
        
    Returns:
        predictions: predicted class (0 or 1)
        probabilities: probability of class 1 (churn)
    """
    if model is None or X is None:
        return None, None
    
    # Predict class
    predictions = model.predict(X)
    
    # Predict probabilities
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)[:, 1]
    else:
        probabilities = None
    
    return predictions, probabilities

def get_feature_importance(model, feature_names):
    """
    Get feature importance from the model
    
    Args:
        model: trained model
        feature_names: list of feature names
        
    Returns:
        feature_importance: pandas DataFrame with feature importance
    """
    if model is None or feature_names is None:
        return None
    
    # Check if model has feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return None
    
    # Ensure feature_names length matches importances length
    if len(feature_names) > len(importances):
        feature_names = feature_names[:len(importances)]
    elif len(feature_names) < len(importances):
        feature_names = feature_names + [f"Unknown_{i}" for i in range(len(importances) - len(feature_names))]
    
    # Create DataFrame
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    return feature_importance

def save_model(model, filename="churn_model.joblib"):
    """
    Save the model to disk
    """
    try:
        joblib.dump(model, filename)
        return True
    except Exception as e:
        st.error(f"Error saving model: {e}")
        return False

def load_model(filename="churn_model.joblib"):
    """
    Load the model from disk
    """
    try:
        model = joblib.load(filename)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
