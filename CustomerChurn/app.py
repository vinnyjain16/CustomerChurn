import streamlit as st
import base64
from pages.home import show_home
from pages.data_upload import show_data_upload
from pages.prediction import show_prediction
from pages.visualization import show_visualization
from pages.feature_importance import show_feature_importance
from pages.database_management import show_database_management
from pages.about import show_about

# Function to create custom header with navigation
def create_custom_header():
    pages = ["Home", "Data Upload", "Prediction", "Visualization", "Feature Importance", "Database Management", "About"]
    current_page = st.session_state.page if "page" in st.session_state else "Home"
    
    # Create header container
    header_container = st.container()
    with header_container:
        cols = st.columns([3, 7])
        with cols[0]:
            st.markdown("""
            <h1 style="margin-bottom:0; font-size:24px;color: #2c3e50;">📊 Customer Churn Predictor</h1>
            """, unsafe_allow_html=True)
        
        with cols[1]:
            # Create horizontal navigation with buttons
            subcols = st.columns(len(pages))
            for i, page in enumerate(pages):
                with subcols[i]:
                    button_style = "background-color:#3498db; color:white;" if page == current_page else "background-color:#f8f9fa; color:#2c3e50;"
                    if st.button(page, key=f"header_nav_{page}", help=f"Go to {page} page", 
                              use_container_width=True):
                        st.session_state.page = page
                        st.rerun()
    
    # Add separator
    st.markdown("<hr style='margin-top:0; margin-bottom:30px;'>", unsafe_allow_html=True)

# Custom CSS to enhance UI
def load_css():
    st.markdown("""
    <style>
         
        
        /* Main area styling */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Headings */
        h1, h2, h3 {
            color: #2c3e50;
            font-family: 'Helvetica Neue', sans-serif;
        }
        h1 {
            font-size: 2.5rem;
            margin-bottom: 1.5rem;
            border-bottom: 2px solid #3498db;
            padding-bottom: 0.5rem;
        }
        h2 {
            font-size: 1.8rem;
            margin-top: 2rem;
            border-left: 4px solid #3498db;
            padding-left: 0.7rem;
        }
        h3 {
            font-size: 1.3rem;
            color: #34495e;
        }
        
        /* Cards and containers */
        .info-card {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
            border-left: 4px solid #3498db;
        }
        
        /* Button styling */
        .stButton>button {
            background-color: #3498db;
            color: white;
            border-radius: 5px;
            border: none;
            padding: 0.5rem 1rem;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #2980b9;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            transform: translateY(-2px);
        }
        
        /* Navigation highlight */
        .navigation-item {
            padding: 0.5rem;
            border-radius: 5px;
            margin-bottom: 0.3rem;
            transition: all 0.2s ease;
        }
        .navigation-item:hover {
            background-color: rgba(52, 152, 219, 0.2);
        }
        .navigation-item.active {
            background-color: rgba(52, 152, 219, 0.3);
            border-left: 3px solid #3498db;
        }
        
        /* Metrics styling */
        div[data-testid="stMetricValue"] {
            font-size: 1.8rem !important;
            color: #2980b9;
        }
        div[data-testid="stMetricLabel"] {
            font-weight: bold;
        }
        
        /* Separator */
        hr {
            margin-top: 2rem;
            margin-bottom: 2rem;
            border: 0;
            height: 1px;
            background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(52, 152, 219, 0.75), rgba(0, 0, 0, 0));
        }
        
        /* Sidebar navigation */
        .sidebar .sidebar-content {
            background-color: #2c3e50;
        }
        section[data-testid="stSidebar"] {
            background-color: #f8f9fa;
            border-right: 1px solid #e9ecef;
        }
        section[data-testid="stSidebar"] h1 {
            color: #2c3e50;
            font-size: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 2px solid #3498db;
        }
        
        /* Dataframe styling */
        .dataframe {
            font-family: 'Helvetica Neue', sans-serif;
            border-collapse: collapse;
            margin: 1rem 0;
            width: 100%;
        }
        .dataframe th {
            background-color: #3498db;
            color: white;
            padding: 0.5rem;
            text-align: left;
            font-weight: 600;
        }
        .dataframe td {
            padding: 0.5rem;
            border-bottom: 1px solid #e9ecef;
        }
        .dataframe tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        .dataframe tr:hover {
            background-color: #e9ecef;
        }
        
        /* Custom tabs styling */
        button[data-baseweb="tab"] {
            background-color: #f8f9fa;
            border-radius: 5px 5px 0 0;
            padding: 0.5rem 1rem;
            margin-right: 0.3rem;
            border: 1px solid #e9ecef;
            border-bottom: none;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            background-color: white;
            border-top: 3px solid #3498db;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
load_css()

# Hide streamlit default menu and footer
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Initialize session state if not exists
if "data" not in st.session_state:
    st.session_state.data = None
if "model" not in st.session_state:
    st.session_state.model = None
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Create custom header with navigation
create_custom_header()

# Sidebar with improved navigation
st.sidebar.title("Navigation")

# Function to handle page navigation
def nav_page(page_name):
    st.session_state.page = page_name
    
# Improved sidebar navigation with styling
pages = ["Home", "Data Upload", "Prediction", "Visualization", "Feature Importance", "Database Management", "About"]
current_page = st.session_state.page if "page" in st.session_state else "Home"

# Create styled navigation buttons
for page_name in pages:
    if current_page == page_name:
        st.sidebar.markdown(f'<div class="navigation-item active">{page_name}</div>', unsafe_allow_html=True)
    else:
        if st.sidebar.button(page_name, key=f"nav_{page_name}"):
            nav_page(page_name)

# Display the selected page
if st.session_state.page == "Home":
    show_home()
elif st.session_state.page == "Data Upload":
    show_data_upload()
elif st.session_state.page == "Prediction":
    show_prediction()
elif st.session_state.page == "Visualization":
    show_visualization()
elif st.session_state.page == "Feature Importance":
    show_feature_importance()
elif st.session_state.page == "Database Management":
    show_database_management()
elif st.session_state.page == "About":
    show_about()

# Enhanced Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="padding: 1rem; background-color: #f1f8ff; border-radius: 10px; border-left: 3px solid #3498db;">
    <h3 style="font-size: 1.1rem; margin-bottom: 0.5rem; color: #2c3e50;">Customer Churn Prediction Tool</h3>
    <p style="font-size: 0.9rem; color: #34495e;">
        A powerful ML-based application to help organizations analyze 
        and predict customer churn patterns.
    </p>
    <p style="font-size: 0.8rem; color: #7f8c8d; margin-top: 0.5rem;">© 2023-2025 All rights reserved</p>
</div>
""", unsafe_allow_html=True)
