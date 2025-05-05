import streamlit as st
import plotly.graph_objects as go
from utils.data_processor import load_sample_data

def show_home():
    """
    Display the home page with information about customer churn prediction
    """
    # Hero section with animation
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; background: linear-gradient(to right, #e0f7fa, #f1f8ff, #e0f7fa); border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: #2c3e50; font-size: 3rem; margin-bottom: 1rem; font-family: 'Helvetica Neue', sans-serif;">
            Customer Churn Prediction and Analysis
        </h1>
        <p style="color: #34495e; font-size: 1.4rem; max-width: 800px; margin: 0 auto 1.5rem auto;">
            Transform your customer retention strategy with advanced AI and machine learning
        </p>
        <div style="max-width: 600px; margin: 0 auto;">
            <div style="height: 5px; background: linear-gradient(to right, #3498db, #2ecc71); border-radius: 5px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Main content with enhanced visuals
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        <div class="info-card" style="background: linear-gradient(to bottom right, #ffffff, #f5f9ff); border-left: 5px solid #3498db;">
        <h2 style="color: #2c3e50; margin-top: 0;">What is Customer Churn?</h2>
        
        <p style="font-size: 1.1rem; line-height: 1.6; color: #2c3e50;">
        Customer churn, or customer attrition, refers to when customers stop doing business with a company. 
        This is a <strong>critical metric</strong> for businesses as acquiring new customers is typically 
        <span style="color: #e74c3c; font-weight: bold;">5-25 times more expensive</span> than retaining existing ones.
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card" style="background: linear-gradient(to bottom right, #ffffff, #f9f5ff); border-left: 5px solid #9b59b6; margin-top: 1.5rem;">
        <h2 style="color: #2c3e50; margin-top: 0;">Why Predict Customer Churn?</h2>
        
        <p style="font-size: 1.05rem; line-height: 1.5;color: #2c3e50;">
        Predicting which customers are likely to churn enables organizations to:
        </p>
        
        <ul style="font-size: 1.05rem; list-style-type: none; padding-left: 0;color: #2c3e50;">
            <li style="margin-bottom: 0.7rem; padding-left: 2rem; position: relative;">
                <span style="position: absolute; left: 0; color: #3498db;">🔔</span>
                <strong>Take proactive measures</strong> to retain valuable customers
            </li>
            <li style="margin-bottom: 0.7rem; padding-left: 2rem; position: relative;">
                <span style="position: absolute; left: 0; color: #3498db;">🔍</span>
                <strong>Identify patterns</strong> in customer behavior that lead to churn
            </li>
            <li style="margin-bottom: 0.7rem; padding-left: 2rem; position: relative;">
                <span style="position: absolute; left: 0; color: #3498db;">🎯</span>
                <strong>Optimize marketing strategies</strong> to target at-risk customers
            </li>
            <li style="margin-bottom: 0.7rem; padding-left: 2rem; position: relative;">
                <span style="position: absolute; left: 0; color: #3498db;">📈</span>
                <strong>Improve overall customer retention</strong> and business profitability
            </li>
            <li style="padding-left: 2rem; position: relative;">
                <span style="position: absolute; left: 0; color: #3498db;">🧠</span>
                <strong>Understand key factors</strong> that influence customer decisions
            </li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Create sample churn visualization using Plotly
        try:
            # Load sample data for visualization
            sample_df = load_sample_data()
            if sample_df is not None:
                # Create churn distribution for demonstration
                churn_counts = sample_df['Churn'].value_counts()
                
                # Create a donut chart
                fig = go.Figure(data=[go.Pie(
                    labels=['Retained', 'Churned'],
                    values=churn_counts.values,
                    hole=.6,
                    marker_colors=['#2ecc71', '#e74c3c'],
                    textinfo='percent+label',
                    pull=[0, 0.1]
                )])
                
                fig.update_layout(
                    title={
                        'text': 'Customer Churn Distribution',
                        'y': 0.95,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top',
                        'font': {'size': 20, 'color': '#2c3e50'}
                    },
                    showlegend=False,
                    height=300,
                    margin=dict(t=50, b=0, l=0, r=0),
                    annotations=[dict(
                        text='Churn<br>Rate',
                        x=0.5, y=0.5,
                        font_size=20,
                        font_color='#34495e',
                        showarrow=False
                    )]
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Key Benefits with enhanced styling
                st.markdown("""
                <h3 style="color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 0.5rem; margin-top: 1rem;">
                    Key Benefits
                </h3>
                """, unsafe_allow_html=True)
                
                benefits = [
                    {
                        "icon": "🔍",
                        "title": "Early Detection",
                        "desc": "Identify at-risk customers before they leave"
                    },
                    {
                        "icon": "📊",
                        "title": "Data-Driven Decisions",
                        "desc": "Make strategic decisions based on accurate predictions"
                    },
                    {
                        "icon": "💡",
                        "title": "Customer Insights",
                        "desc": "Understand what drives customer loyalty and attrition"
                    },
                    {
                        "icon": "💰",
                        "title": "ROI Improvement",
                        "desc": "Reduce churn to improve overall business performance"
                    }
                ]
                
                for benefit in benefits:
                    st.markdown(f"""
                    <div style="background-color: #f8f9fa; border-radius: 8px; padding: 0.7rem; margin-bottom: 0.7rem; 
                         border-left: 3px solid #3498db; display: flex; align-items: center;">
                        <div style="font-size: 1.5rem; margin-right: 0.7rem; min-width: 40px; text-align: center;">
                            {benefit["icon"]}
                        </div>
                        <div>
                            <h4 style="margin: 0; color: #2c3e50; font-size: 1rem;">{benefit["title"]}</h4>
                            <p style="margin: 0; font-size: 0.9rem; color: #7f8c8d;">{benefit["desc"]}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Could not load visualization: {e}")
    
    # How it works section with enhanced styling
    st.markdown("""
    <div style="background: linear-gradient(to right, #f5f9ff, #ffffff, #f5f9ff); padding: 2rem; border-radius: 10px; margin: 2rem 0;">
        <h2 style="text-align: center; color: #2c3e50; margin-top: 0; border-bottom: none;">
            How It Works
        </h2>
        <div style="width: 100px; height: 3px; background: linear-gradient(to right, #3498db, #2ecc71); margin: 0 auto 2rem auto; border-radius: 3px;"></div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    steps = [
        {"icon": "📤", "title": "Upload Data", "desc": "Upload your customer data or use our sample dataset"},
        {"icon": "📊", "title": "Analyze", "desc": "Explore patterns and correlations in your data"},
        {"icon": "🧠", "title": "Train Model", "desc": "Train and evaluate machine learning models"},
        {"icon": "🔮", "title": "Predict", "desc": "Get churn predictions and actionable insights"}
    ]
    
    for i, (col, step) in enumerate(zip([col1, col2, col3, col4], steps)):
        with col:
            st.markdown(f"""
            <div style="text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem; color: #3498db;">
                    {step["icon"]}
                </div>
                <h3 style="margin: 0 0 0.5rem 0; color: #2c3e50; font-size: 1.1rem;">
                    {i+1}. {step["title"]}
                </h3>
                <p style="font-size: 0.9rem; color: #7f8c8d; margin: 0;">
                    {step["desc"]}
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Call to action with enhanced styling
    st.markdown("""
    <div style="background: linear-gradient(135deg, #3498db, #2c3e50); color: white; padding: 2rem; border-radius: 10px; text-align: center; margin: 2rem 0;">
        <h2 style="color: white; margin-top: 0; border-bottom: none;">Ready to reduce customer churn?</h2>
        <p style="font-size: 1.1rem; max-width: 600px; margin: 1rem auto;">
            Start analyzing your customer data and predicting churn with our powerful tools
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Get Started with Sample Data", key="sample_data_btn", help="Load our sample dataset to explore the tool"):
            # Use session state to navigate to data upload page
            if "data" not in st.session_state or st.session_state.data is None:
                sample_df = load_sample_data()
                if sample_df is not None:
                    st.session_state.data = sample_df
                    st.success("Sample data loaded successfully!")
            
            st.session_state.page = "Data Upload"
            st.rerun()
    
    with col2:
        if st.button("📤 Upload Your Own Data", key="upload_data_btn", help="Upload your own dataset for analysis"):
            # Use session state to navigate to data upload page
            st.session_state.page = "Data Upload"
            st.rerun()
            
    # Social proof section
    st.markdown("""
    <div style="margin: 3rem 0; text-align: center;">
        <h3 style="color: #2c3e50; margin-bottom: 1.5rem;">Trusted by Industry Leaders</h3>
        <div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 1rem; margin: 0 auto; max-width: 800px;">
            <div style="text-align: center; padding: 1rem;">
                <div style="font-size: 1.2rem; font-weight: bold; color: #3498db;">TelcoMax</div>
                <div style="color: #7f8c8d;">Reduced churn by 25%</div>
            </div>
            <div style="text-align: center; padding: 1rem;">
                <div style="font-size: 1.2rem; font-weight: bold; color: #3498db;">FinServe Pro</div>
                <div style="color: #7f8c8d;">Improved retention by 32%</div>
            </div>
            <div style="text-align: center; padding: 1rem;">
                <div style="font-size: 1.2rem; font-weight: bold; color: #3498db;">RetailGiant</div>
                <div style="color: #7f8c8d;">23% ROI increase</div>
            </div>
            <div style="text-align: center; padding: 1rem;">
                <div style="font-size: 1.2rem; font-weight: bold; color: #3498db;">HealthPlus</div>
                <div style="color: #7f8c8d;">45% better targeting</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
