import streamlit as st
import pandas as pd
import datetime
from utils.data_processor import load_sample_data, process_uploaded_file
from utils.database import load_data_from_db, get_model_performance_history

def show_data_upload():
    """
    Display the data upload page
    """
    st.title("Data Upload & Exploration")
    st.markdown("""
    <div class="info-card; style="color: #000000;">
        Upload your customer data, use our sample dataset, or retrieve data from the database to get started.
    </div>
    """, unsafe_allow_html=True)
    
    # Show dataset requirements information
    with st.expander("📋 Dataset Requirements for Churn Prediction", expanded=True):
        st.markdown("""
        
        ### Required Data Format for Churn Prediction
        
        Your dataset should ideally contain these types of columns:
        
        1. **Customer Identifier**: A unique ID for each customer (e.g., `customer_id`, `user_id`)
        
        2. **Target Variable**: Column indicating whether a customer has churned
           - Column named `Churn` or similar
           - Values should be `Yes`/`No`, `True`/`False`, or `1`/`0`
        
        3. **Customer Demographics**: Information about the customer
           - Gender, age, or senior status
           - Relationship status (partner, dependents)
           - Tenure (how long they've been a customer)
        
        4. **Service Usage**: What services they use
           - Types of services subscribed to
           - Service features used
        
        5. **Account Information**: Details about their account
           - Contract type
           - Payment method
           - Billing preferences
        
        6. **Financial Information**: Spending data
           - Monthly charges
           - Total charges
        
        **File Format**: CSV or Excel files (.csv, .xlsx, .xls)
        
        **Example Dataset**: Our sample dataset (telco_customer_churn.csv) contains all required fields for analyzing customer churn in a telecommunications company.
        """)
    
    st.markdown("---")
    
    # Data upload options
    upload_option = st.radio(
        "Choose an option:",
        ["Upload my own data", "Use sample data", "Load from database"]
    )
    
    if upload_option == "Upload my own data":
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload a CSV or Excel file:",
            type=["csv", "xlsx", "xls"],
            help="Make sure your data includes customer attributes and a churn indicator column."
        )
        
        # Option to save to database
        save_to_db = st.checkbox("Save to database for future use", 
                               value=True, 
                               help="Store this data in the database for future sessions")
        
        if uploaded_file is not None:
            # Process the uploaded file
            df = process_uploaded_file(uploaded_file, save_to_db=save_to_db)
            if df is not None:
                st.session_state.data = df
                st.success(f"Data uploaded successfully! {len(df)} rows and {len(df.columns)} columns detected.")
                
                # Add timestamp to session state
                st.session_state.upload_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    elif upload_option == "Use sample data":
        # Option to save to database
        save_to_db = st.checkbox("Save to database for future use", 
                               value=True, 
                               help="Store sample data in the database for future sessions")
        
        if st.button("Load Sample Data"):
            # Load the sample data
            df = load_sample_data(use_db=save_to_db)
            if df is not None:
                st.session_state.data = df
                st.success(f"Sample data loaded successfully! {len(df)} rows and {len(df.columns)} columns.")
                
                # Add timestamp to session state
                st.session_state.upload_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    else:  # Load from database
        st.markdown("""
        <div class="info-card" style="background-color: #f0f7ff;">
            <h3 style="margin-top: 0;">Database Data</h3>
            <p>Load previously uploaded customer data from the database.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Check if there's data in the database
        try:
            # Try to load a small sample to check if data exists
            test_df = load_data_from_db('customers', limit=1)
            
            if test_df is not None and not test_df.empty:
                # Data exists, offer to load it
                row_limit = st.slider("Maximum number of rows to load:", 100, 10000, 1000, 100)
                
                if st.button(f"Load {row_limit} rows from database"):
                    df = load_data_from_db('customers', limit=row_limit)
                    if df is not None and not df.empty:
                        st.session_state.data = df
                        st.success(f"Data loaded from database! {len(df)} rows and {len(df.columns)} columns retrieved.")
                        
                        # Add timestamp to session state
                        st.session_state.upload_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state.data_source = "Database"
                    else:
                        st.error("Failed to load data from database.")
            else:
                st.warning("No data found in the database. Please upload data first or use the sample dataset.")
                
        except Exception as e:
            st.error(f"Error connecting to database: {str(e)}")
            st.info("Please upload your own data or use the sample dataset instead.")
    
    # Data preview and exploration
    if "data" in st.session_state and st.session_state.data is not None:
        df = st.session_state.data
        
        # Data Overview
        st.markdown("## Data Overview")
        
        # Show sample of the data
        st.markdown("### Data Preview")
        st.dataframe(df.head(10))
        
        # Data shape
        st.markdown(f"**Dataset shape:** {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Column information
        st.markdown("### Column Information")
        
        # Create a dataframe to display column info
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str),
            'Missing Values': df.isnull().sum(),
            'Missing (%)': round(df.isnull().sum() / len(df) * 100, 2),
            'Unique Values': [df[col].nunique() for col in df.columns]
        })
        
        st.dataframe(col_info)
        
        # Data Statistics
        st.markdown("### Data Statistics")
        
        # Create tabs for different types of statistics
        tab1, tab2 = st.tabs(["Numerical", "Categorical"])
        
        with tab1:
            # Numeric columns
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if not numeric_cols.empty:
                st.dataframe(df[numeric_cols].describe())
            else:
                st.info("No numerical columns found in the dataset.")
        
        with tab2:
            # Categorical columns
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            if not cat_cols.empty:
                for col in cat_cols:
                    st.markdown(f"**{col}** - Top 10 values:")
                    st.dataframe(df[col].value_counts().head(10).reset_index().rename(
                        columns={"index": col, col: "Count"}
                    ))
            else:
                st.info("No categorical columns found in the dataset.")
        
        # Target column selection for churn prediction
        st.markdown("## Target Column Selection")
        st.info("Select the column that indicates customer churn (e.g., 'Churn', 'Attrition', etc.)")
        
        target_options = ["Select a column"] + list(df.columns)
        target_column = st.selectbox("Select your target column:", target_options)
        
        if target_column != "Select a column":
            # Display distribution of target variable
            st.markdown(f"### Distribution of {target_column}")
            
            value_counts = df[target_column].value_counts()
            st.bar_chart(value_counts)
            
            # Store target column in session state
            st.session_state.target_column = target_column
            
            # Proceed to modeling
            st.markdown("## Next Steps")
            if st.button("Proceed to Prediction"):
                st.session_state.page = "Prediction"
                st.rerun()
    else:
        st.info("Please upload a file or load the sample data to continue.")
