import os
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import datetime
import json

# Create database connection
DATABASE_URL = "postgresql://postgres:Vinny%401998@localhost:5432/customerchurn"

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=3600)
Base = declarative_base()
Session = sessionmaker(bind=engine, expire_on_commit=False)

class Customer(Base):
    """Customer data table"""
    __tablename__ = 'customers'
    
    id = Column(Integer, primary_key=True)
    customer_id = Column(String(255), unique=True)
    gender = Column(String(50), nullable=True)
    senior_citizen = Column(Boolean, nullable=True)
    partner = Column(Boolean, nullable=True)
    dependents = Column(Boolean, nullable=True)
    tenure = Column(Integer, nullable=True)
    phone_service = Column(Boolean, nullable=True)
    multiple_lines = Column(String(50), nullable=True)
    internet_service = Column(String(50), nullable=True)
    online_security = Column(String(50), nullable=True)
    online_backup = Column(String(50), nullable=True)
    device_protection = Column(String(50), nullable=True)
    tech_support = Column(String(50), nullable=True)
    streaming_tv = Column(String(50), nullable=True)
    streaming_movies = Column(String(50), nullable=True)
    contract = Column(String(50), nullable=True)
    paperless_billing = Column(Boolean, nullable=True)
    payment_method = Column(String(50), nullable=True)
    monthly_charges = Column(Float, nullable=True)
    total_charges = Column(Float, nullable=True)
    churn = Column(Boolean, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    predictions = relationship("ChurnPrediction", back_populates="customer", cascade="all, delete-orphan")
    
    def to_dict(self):
        return {
            'id': self.id,
            'customer_id': self.customer_id,
            'gender': self.gender,
            'senior_citizen': self.senior_citizen,
            'partner': self.partner,
            'dependents': self.dependents,
            'tenure': self.tenure,
            'phone_service': self.phone_service,
            'multiple_lines': self.multiple_lines,
            'internet_service': self.internet_service,
            'online_security': self.online_security,
            'online_backup': self.online_backup,
            'device_protection': self.device_protection,
            'tech_support': self.tech_support,
            'streaming_tv': self.streaming_tv,
            'streaming_movies': self.streaming_movies,
            'contract': self.contract,
            'paperless_billing': self.paperless_billing,
            'payment_method': self.payment_method,
            'monthly_charges': self.monthly_charges,
            'total_charges': self.total_charges,
            'churn': self.churn,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class ChurnPrediction(Base):
    """Churn prediction results table"""
    __tablename__ = 'churn_predictions'
    
    id = Column(Integer, primary_key=True)
    customer_id = Column(Integer, ForeignKey('customers.id'))
    prediction = Column(Boolean)
    probability = Column(Float)
    model_type = Column(String(100))
    model_version = Column(String(50))
    feature_importance = Column(Text, nullable=True)  # Stored as JSON
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    customer = relationship("Customer", back_populates="predictions")
    
    def to_dict(self):
        return {
            'id': self.id,
            'customer_id': self.customer_id,
            'prediction': self.prediction,
            'probability': self.probability,
            'model_type': self.model_type,
            'model_version': self.model_version,
            'feature_importance': json.loads(self.feature_importance) if self.feature_importance else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class MLModel(Base):
    """ML model metadata table"""
    __tablename__ = 'ml_models'
    
    id = Column(Integer, primary_key=True)
    model_type = Column(String(100))
    model_version = Column(String(50))
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    training_date = Column(DateTime, default=datetime.datetime.utcnow)
    model_params = Column(Text)  # Stored as JSON
    
    def to_dict(self):
        return {
            'id': self.id,
            'model_type': self.model_type,
            'model_version': self.model_version,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'training_date': self.training_date.isoformat() if self.training_date else None,
            'model_params': json.loads(self.model_params) if self.model_params else None
        }

class DatasetUpload(Base):
    """Dataset upload metadata table"""
    __tablename__ = 'dataset_uploads'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255))
    rows = Column(Integer)
    columns = Column(Integer)
    upload_date = Column(DateTime, default=datetime.datetime.utcnow)
    description = Column(Text, nullable=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.filename,
            'rows': self.rows,
            'columns': self.columns,
            'upload_date': self.upload_date.isoformat() if self.upload_date else None,
            'description': self.description
        }

# Database initialization function
def init_db():
    Base.metadata.create_all(engine)

# Helper functions
def get_session():
    """Get a new database session"""
    return Session()

def save_dataframe_to_db(df, table_name=None):
    """Save a pandas DataFrame to the database"""
    if table_name is None:
        table_name = 'customers'
    
    conn = None
    try:
        # Create a copy of the dataframe to avoid modifying the original
        df_copy = df.copy()
        
        # Map column names to match database schema for customers table
        if table_name == 'customers':
            # Standardize column names - convert to lowercase and replace spaces with underscores
            df_copy.columns = [col.lower().replace(' ', '_') for col in df_copy.columns]
            
            # Map common column name variations to our schema
            column_mapping = {
                'user_id': 'customer_id',
                'product_id': 'customer_id',
                'customerid': 'customer_id',
                'age': 'senior_citizen',  # We'll convert age to senior_citizen later if needed
                'phoneservice': 'phone_service',
                'multiplelines': 'multiple_lines',
                'internetservice': 'internet_service',
                'onlinesecurity': 'online_security',
                'onlinebackup': 'online_backup',
                'deviceprotection': 'device_protection',
                'techsupport': 'tech_support',
                'streamingtv': 'streaming_tv',
                'streamingmovies': 'streaming_movies',
                'paperlessbilling': 'paperless_billing',
                'paymentmethod': 'payment_method',
                'monthlycharges': 'monthly_charges',
                'totalcharges': 'total_charges'
            }
            
            # Rename columns based on mapping
            df_copy = df_copy.rename(columns=column_mapping)
            
            # Ensure only valid columns are kept
            valid_columns = [
                'customer_id', 'gender', 'senior_citizen', 'partner', 'dependents', 
                'tenure', 'phone_service', 'multiple_lines', 'internet_service',
                'online_security', 'online_backup', 'device_protection', 'tech_support',
                'streaming_tv', 'streaming_movies', 'contract', 'paperless_billing',
                'payment_method', 'monthly_charges', 'total_charges', 'churn'
            ]
            
            # Keep only columns that exist in our schema
            existing_valid_columns = [col for col in valid_columns if col in df_copy.columns]
            df_copy = df_copy[existing_valid_columns]
        
        # Convert Yes/No to boolean for relevant columns
        bool_columns = ['churn', 'partner', 'dependents', 'phone_service', 'paperless_billing']
        for col in bool_columns:
            if col in df_copy.columns:
                # Handle different variations of Yes/No
                if df_copy[col].dtype == 'object':
                    df_copy[col] = df_copy[col].str.lower().map({'yes': True, 'no': False, 'y': True, 'n': False, 'true': True, 'false': False})
        
        # Handle SeniorCitizen special case
        if 'senior_citizen' in df_copy.columns:
            # If it contains 0/1 values
            if set(df_copy['senior_citizen'].unique()).issubset({0, 1}):
                df_copy['senior_citizen'] = df_copy['senior_citizen'].astype(bool)
            # If it contains Yes/No values
            elif df_copy['senior_citizen'].dtype == 'object':
                df_copy['senior_citizen'] = df_copy['senior_citizen'].str.lower().map({'yes': True, 'no': False, 'y': True, 'n': False})
            # If it contains age values (assume senior is 65+)
            elif df_copy['senior_citizen'].dtype in ['int64', 'float64'] and df_copy['senior_citizen'].max() > 1:
                df_copy['senior_citizen'] = df_copy['senior_citizen'] >= 65
        
        # Get a new connection with AUTOCOMMIT
        conn = engine.connect().execution_options(isolation_level="AUTOCOMMIT")
        
        # Save to database
        df_copy.to_sql(table_name, conn, if_exists='append', index=False)
        return True, f"Successfully saved {len(df_copy)} rows to {table_name}"
    except Exception as e:
        return False, f"Error saving data to database: {str(e)}"
    finally:
        if conn:
            conn.close()

def load_data_from_db(table_name, limit=1000):
    """Load data from the database into a pandas DataFrame"""
    conn = None
    try:
        # Create a new connection for this query to avoid transaction issues
        conn = engine.connect().execution_options(isolation_level="AUTOCOMMIT")
        
        # Handle case when limit is None
        limit_clause = "" if limit is None else f"LIMIT {limit}"
        
        # Safely format table name to prevent SQL injection
        # For a more robust solution, you'd use a parameterized query system that handles tables safely
        table_name_safe = ''.join(c for c in table_name if c.isalnum() or c == '_')
        if table_name_safe != table_name:
            st.warning(f"Invalid table name detected: using '{table_name_safe}' instead")
            
        query = f"SELECT * FROM {table_name_safe} {limit_clause}"
        
        # Use the connection directly for better control
        df = pd.read_sql(query, conn)
        
        # Convert boolean columns back to Yes/No for display purposes
        bool_columns = ['churn', 'partner', 'dependents', 'phone_service', 'paperless_billing', 'senior_citizen']
        for col in bool_columns:
            if col in df.columns:
                df[col] = df[col].map({True: 'Yes', False: 'No'})
        
        return df
    except Exception as e:
        st.error(f"Error loading data from database: {str(e)}")
        return None
    finally:
        if conn:
            conn.close()

def save_prediction_to_db(customer_id, prediction, probability, model_type, model_version, feature_importance=None):
    """Save a prediction to the database"""
    session = None
    try:
        session = get_session()
        
        # Convert feature importance to JSON string if provided
        feature_importance_json = None
        if feature_importance is not None:
            feature_importance_json = json.dumps(feature_importance.to_dict())
        
        # Create new prediction record
        new_prediction = ChurnPrediction(
            customer_id=customer_id,
            prediction=prediction,
            probability=probability,
            model_type=model_type,
            model_version=model_version,
            feature_importance=feature_importance_json
        )
        
        session.add(new_prediction)
        session.commit()
        prediction_id = new_prediction.id
        session.close()
        
        return True, prediction_id
    except Exception as e:
        if session:
            session.rollback()
            session.close()
        return False, str(e)

def save_model_metadata(model_type, model_version, metrics, model_params):
    """Save model metadata to the database"""
    session = None
    try:
        session = get_session()
        
        # Convert model parameters to JSON string
        model_params_json = json.dumps(model_params)
        
        # Create new model record
        new_model = MLModel(
            model_type=model_type,
            model_version=model_version,
            accuracy=metrics.get('accuracy', 0),
            precision=metrics.get('precision', 0),
            recall=metrics.get('recall', 0),
            f1_score=metrics.get('f1', 0),
            model_params=model_params_json
        )
        
        session.add(new_model)
        session.commit()
        model_id = new_model.id
        session.close()
        
        return True, model_id
    except Exception as e:
        if session:
            session.rollback()
            session.close()
        return False, str(e)

def log_dataset_upload(filename, rows, columns, description=None):
    """Log a dataset upload to the database"""
    session = None
    try:
        session = get_session()
        
        # Create new dataset upload record
        new_upload = DatasetUpload(
            filename=filename,
            rows=rows,
            columns=columns,
            description=description
        )
        
        session.add(new_upload)
        session.commit()
        upload_id = new_upload.id
        session.close()
        
        return True, upload_id
    except Exception as e:
        if session:
            session.rollback()
            session.close()
        return False, str(e)

def get_recent_predictions(limit=10):
    """Get recent predictions from the database"""
    conn = None
    try:
        # Create a new connection with AUTOCOMMIT to avoid transaction issues
        conn = engine.connect().execution_options(isolation_level="AUTOCOMMIT")
        
        # Use PostgreSQL's specific parameter syntax with number
        query = f"""
        SELECT cp.id, c.customer_id, cp.prediction, cp.probability, 
               cp.model_type, cp.created_at
        FROM churn_predictions cp
        JOIN customers c ON cp.customer_id = c.id
        ORDER BY cp.created_at DESC
        LIMIT {limit}
        """
        
        # Execute query without parameters (limit is directly in the query)
        df = pd.read_sql(query, conn)
        
        # Convert boolean to Yes/No for display
        if 'prediction' in df.columns:
            df['prediction'] = df['prediction'].map({True: 'Yes', False: 'No'})
            
        return df
    except Exception as e:
        st.error(f"Error loading recent predictions: {str(e)}")
        return None
    finally:
        if conn:
            conn.close()

def get_model_performance_history():
    """Get model performance history from the database"""
    conn = None
    try:
        # Create a new connection with AUTOCOMMIT to avoid transaction issues
        conn = engine.connect().execution_options(isolation_level="AUTOCOMMIT")
        
        query = """
        SELECT model_type, model_version, accuracy, precision, recall, f1_score, training_date
        FROM ml_models
        ORDER BY training_date DESC
        """
        
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        st.error(f"Error loading model performance history: {str(e)}")
        return None
    finally:
        if conn:
            conn.close()

# Initialize the database
init_db()