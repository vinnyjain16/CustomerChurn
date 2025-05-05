import streamlit as st

def show_about():
    """
    Display the about page with information about the application
    """
    st.title("About the Customer Churn Prediction Tool")
    
    st.markdown("""
    ## Overview
    
    The Customer Churn Prediction Tool is a powerful application designed to help businesses identify
    customers who are likely to churn (discontinue their service). By leveraging machine learning algorithms,
    the tool analyzes historical customer data to predict future churn behaviors, enabling proactive 
    retention strategies.
    
    ## Key Features
    
    - **Data Analysis**: Explore and analyze customer data to understand patterns and trends
    - **Machine Learning Models**: Train multiple models to predict customer churn
    - **Feature Importance**: Identify the key factors that influence customer churn
    - **Visualization**: Interactive charts and graphs to visualize insights
    - **Risk Assessment**: Identify high-risk customers likely to churn
    - **Downloadable Reports**: Export predictions and insights for further action
    
    ## Technologies Used
    
    This application is built using state-of-the-art technologies:
    
    - **Streamlit**: For the user interface and web application
    - **Pandas & NumPy**: For data manipulation and analysis
    - **Scikit-learn**: For machine learning models and algorithms
    - **Plotly**: For interactive data visualizations
    - **XGBoost**: For advanced gradient boosting algorithms
    
    ## Machine Learning Models
    
    The application offers several machine learning models for churn prediction:
    
    - **Logistic Regression**: A straightforward model for binary classification
    - **Random Forest**: An ensemble method that builds multiple decision trees
    - **Gradient Boosting**: Sequential building of trees to correct errors from previous trees
    - **XGBoost**: An optimized gradient boosting implementation with enhanced performance
    
    ## How to Use This Tool
    
    1. **Upload Data**: Start by uploading your customer data or using our sample dataset
    2. **Explore Data**: Visualize and understand your customer data
    3. **Train Models**: Select and train machine learning models
    4. **Generate Predictions**: Predict which customers are at risk of churning
    5. **Analyze Results**: Explore feature importance and risk factors
    6. **Take Action**: Use the insights to develop targeted retention strategies
    
    ## Best Practices for Customer Retention
    
    Based on common findings from churn analysis, here are some best practices:
    
    - **Proactive Engagement**: Reach out to high-risk customers before they decide to leave
    - **Personalized Offers**: Create tailored incentives based on customer preferences
    - **Feedback Collection**: Regularly collect and act on customer feedback
    - **Service Improvement**: Address common pain points identified in churn analysis
    - **Loyalty Programs**: Reward long-term customers with special benefits
    
    ## Resources
    
    To learn more about customer churn prediction and retention strategies:
    
    - [Harvard Business Review: The Value of Keeping the Right Customers](https://hbr.org/2014/10/the-value-of-keeping-the-right-customers)
    - [Toward Data Science: Customer Churn Prediction](https://towardsdatascience.com/customer-churn-prediction-15d4a6cc7da3)
    - [Forbes: Customer Retention Strategies](https://www.forbes.com/sites/jiawertz/2018/09/12/dont-spend-5-times-more-attracting-new-customers-nurture-the-existing-ones/)
    """)
    
    # Contact Information
    st.markdown("---")
    st.markdown("## Contact Information")
    
    st.markdown("""
    For questions, feedback, or support, please contact:
    
    - **Email**: support@churnpredictor.example.com
    - **Website**: www.churnpredictor.example.com
    
    © 2023 Customer Churn Prediction Tool. All rights reserved.
    """)
