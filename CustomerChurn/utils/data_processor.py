import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import streamlit as st
import io
from utils.database import save_dataframe_to_db, load_data_from_db, log_dataset_upload

def load_sample_data(use_db=False):
    """
    Load the sample telco customer churn dataset
    
    Args:
        use_db: Whether to load from database if available
    """
    try:
        # Try to load from database first if requested
        if use_db:
            db_data = load_data_from_db('customers')
            if db_data is not None and not db_data.empty:
                st.success("Loaded sample data from database")
                return db_data
        
        # Fall back to file if not in database or not requested from db
        sample_df = pd.read_csv("sample_data/telco_customer_churn.csv")
        
        # Save to database if requested and not already there
        if use_db:
            success, message = save_dataframe_to_db(sample_df)
            if success:
                st.success("Sample data saved to database")
                
                # Log the upload
                log_dataset_upload(
                    filename="telco_customer_churn.csv",
                    rows=len(sample_df),
                    columns=len(sample_df.columns),
                    description="Sample dataset of telecom customer churn data"
                )
            else:
                st.warning(f"Note: {message}")
        
        return sample_df
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        return None

def process_uploaded_file(uploaded_file, save_to_db=False):
    """
    Process the uploaded file and return a pandas DataFrame
    
    Args:
        uploaded_file: The uploaded file object
        save_to_db: Whether to save the data to the database
    """
    try:
        # Check file type
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
        
        # Process column names for consistency
        df.columns = [col.strip().replace(' ', '_') for col in df.columns]
        
        # Save to database if requested
        if save_to_db and df is not None:
            success, message = save_dataframe_to_db(df)
            if success:
                st.success(message)
                
                # Log the upload
                log_dataset_upload(
                    filename=uploaded_file.name,
                    rows=len(df),
                    columns=len(df.columns),
                    description=f"Uploaded by user on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
            else:
                st.warning(f"Note: {message}")
        
        return df
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

def preprocess_data(df, target_column=None):
    """
    Preprocess the data for machine learning
    
    Args:
        df: pandas DataFrame
        target_column: name of the target column (if None, assumes no target column)
        
    Returns:
        X: preprocessed features
        y: target variable (if target_column is provided)
        preprocessor: the fitted preprocessor object
    """
    if df is None:
        return None, None, None
    
    # Make a copy to avoid modifying the original
    data = df.copy()
    
    # Basic cleaning
    data = data.replace('', np.nan)
    
    # Extract target if specified
    y = None
    if target_column and target_column in data.columns:
        # Handle NaN values in target column
        if data[target_column].isna().any():
            st.warning(f"Found {data[target_column].isna().sum()} missing values in target column. Dropping these rows.")
            data = data.dropna(subset=[target_column])
        
        # Convert Yes/No to 1/0 if needed
        if data[target_column].dtype == 'object':
            y_mapping = {'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, 'Y': 1, 'N': 0, 'TRUE': 1, 'FALSE': 0, 'True': 1, 'False': 0}
            y = data[target_column].map(y_mapping)
            if y.isna().any():  # If any values couldn't be mapped
                st.error(f"Target column contains values other than Yes/No. Please verify your data.")
                return None, None, None
        else:
            y = data[target_column]
        
        data = data.drop(columns=[target_column])
    
    # Identify numeric and categorical columns
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Apply preprocessing
    X = preprocessor.fit_transform(data)
    
    return X, y, preprocessor

def get_feature_names_from_preprocessor(preprocessor, original_df):
    """
    Get feature names after preprocessing
    """
    # Get all transformer names
    feature_names = []
    
    # Get original column names
    numeric_cols = original_df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = original_df.select_dtypes(include=['object', 'category']).columns
    
    # Handle numeric features (which keep their original names)
    for col in numeric_cols:
        feature_names.append(col)
    
    # Handle categorical features (which are one-hot encoded)
    for col in categorical_cols:
        unique_values = original_df[col].dropna().unique()
        for value in unique_values:
            feature_names.append(f"{col}_{value}")
    
    return feature_names

def export_results(df, predictions, format='csv'):
    """
    Export the results to a downloadable file
    """
    # Create a copy of the original dataframe
    result_df = df.copy()
    
    # Add predictions column
    result_df['Predicted_Churn'] = predictions
    
    # Prepare for download
    if format == 'csv':
        output = io.BytesIO()
        result_df.to_csv(output, index=False)
        return output.getvalue()
    elif format == 'excel':
        output = io.BytesIO()
        result_df.to_excel(output, index=False)
        return output.getvalue()
    else:
        raise ValueError(f"Unsupported format: {format}")
