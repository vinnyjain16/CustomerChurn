import streamlit as st
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils.database import (
    get_session, load_data_from_db, get_recent_predictions, 
    get_model_performance_history
)
import sqlalchemy as sa
from dotenv import load_dotenv
load_dotenv()

def show_database_management():
    """
    Display the database management page
    """
    st.title("Database Management")
    
    st.markdown("""
    <div class="info-card" style="background: linear-gradient(to bottom right, #ffffff, #f0f7ff);">
        <h3 style="margin-top: 0;color: #2c3e50;">Database Overview</h3>
        <p>View and manage your database content, including customer data, models, and predictions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different database sections
    tabs = st.tabs(["Customer Data", "Model Performance", "Predictions", "Advanced"])
    
    # Tab 1: Customer Data
    with tabs[0]:
        st.markdown("## Customer Data")
        
        try:
            # Check if customers table exists and has data
            customers_df = load_data_from_db('customers', limit=5)
            
            if customers_df is not None and not customers_df.empty:
                # Show customer data overview
                try:
                    # Get total count using SQL query instead of loading all data
                    engine = sa.create_engine(st.secrets["connections"]["postgresql"])
                    with engine.connect() as conn:
                        result = conn.execute(sa.text("SELECT COUNT(*) FROM customers"))
                        total_customers = result.scalar()
                except Exception as e:
                    st.error(f"Error getting customer count: {str(e)}")
                    total_customers = len(customers_df)  # Fallback to sample count
                
                # Get churn distribution if Churn column exists
                churn_distribution = None
                if 'churn' in customers_df.columns:
                    churn_query = """
                    SELECT churn, COUNT(*) as count
                    FROM customers
                    GROUP BY churn
                    """
                    try:
                        engine = sa.create_engine(st.secrets["connections"]["postgresql"])
                        churn_distribution = pd.read_sql(churn_query, engine)
                    except Exception as e:
                        st.error(f"Error querying churn distribution: {str(e)}")
                
                # Display metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Customers", f"{total_customers:,}")
                
                with col2:
                    if churn_distribution is not None:
                        # Calculate churn rate
                        churn_count = churn_distribution.loc[churn_distribution['churn'] == True, 'count'].sum()
                        churn_rate = churn_count / total_customers if total_customers > 0 else 0
                        st.metric("Churn Rate", f"{churn_rate:.2%}")
                
                # Show sample of customer data
                st.markdown("### Sample Customer Records")
                st.dataframe(customers_df)
                
                # Option to view more data
                if st.button("View more customer data"):
                    more_customers = load_data_from_db('customers', limit=100)
                    st.dataframe(more_customers)
                
                # Churn distribution visualization if available
                if churn_distribution is not None:
                    st.markdown("### Churn Distribution")
                    fig = px.pie(
                        churn_distribution, 
                        values='count', 
                        names='churn',
                        color='churn',
                        color_discrete_map={True: '#e74c3c', False: '#2ecc71'},
                        title="Customer Churn Distribution"
                    )
                    fig.update_traces(
                        textinfo='percent+label', 
                        pull=[0.05, 0],
                        marker=dict(line=dict(color='white', width=2))
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No customer data found in the database.")
                st.info("Upload data in the 'Data Upload' page to populate the database.")
        
        except Exception as e:
            st.error(f"Error accessing customer data: {str(e)}")
    
    # Tab 2: Model Performance
    with tabs[1]:
        st.markdown("## Model Performance History")
        
        try:
            # Get model performance history
            model_history = get_model_performance_history()
            
            if model_history is not None and not model_history.empty:
                # Display model history table
                st.markdown("### Model Training History")
                
                # Format the table for display
                display_df = model_history.copy()
                
                # Format the training date
                if 'training_date' in display_df.columns:
                    display_df['training_date'] = pd.to_datetime(display_df['training_date']).dt.strftime('%Y-%m-%d %H:%M')
                
                # Format metric columns as percentages
                metric_cols = ['accuracy', 'precision', 'recall', 'f1_score']
                for col in metric_cols:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A")
                
                st.dataframe(display_df)
                
                # Visualize model performance trends
                if len(model_history) > 1 and all(col in model_history.columns for col in metric_cols):
                    st.markdown("### Performance Trends")
                    
                    # Prepare data for visualization
                    plot_df = model_history.copy()
                    plot_df['training_date'] = pd.to_datetime(plot_df['training_date'])
                    plot_df = plot_df.sort_values('training_date')
                    
                    # Create trend visualization
                    fig = go.Figure()
                    
                    for metric in metric_cols:
                        fig.add_trace(go.Scatter(
                            x=plot_df['training_date'],
                            y=plot_df[metric],
                            mode='lines+markers',
                            name=metric.replace('_', ' ').title()
                        ))
                    
                    # Update layout
                    fig.update_layout(
                        title="Model Performance Metrics Over Time",
                        xaxis_title="Training Date",
                        yaxis_title="Metric Value",
                        yaxis=dict(tickformat='.0%'),
                        legend_title="Metrics",
                        hovermode="x unified"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Model comparison by type
                    st.markdown("### Model Type Comparison")
                    
                    if 'model_type' in model_history.columns:
                        # Group by model type and calculate average metrics
                        model_comparison = model_history.groupby('model_type')[metric_cols].mean().reset_index()
                        
                        # Create grouped bar chart
                        fig2 = px.bar(
                            model_comparison.melt(id_vars=['model_type'], value_vars=metric_cols),
                            x='model_type',
                            y='value',
                            color='variable',
                            barmode='group',
                            title="Average Performance by Model Type",
                            labels={
                                'model_type': 'Model Type',
                                'value': 'Metric Value',
                                'variable': 'Metric'
                            }
                        )
                        
                        # Update layout
                        fig2.update_layout(
                            yaxis=dict(tickformat='.0%'),
                            xaxis_title="Model Type",
                            yaxis_title="Average Metric Value"
                        )
                        
                        st.plotly_chart(fig2, use_container_width=True)
            else:
                st.warning("No model performance history found in the database.")
                st.info("Train models in the 'Prediction' page to populate model performance data.")
        
        except Exception as e:
            st.error(f"Error accessing model performance data: {str(e)}")
    
    # Tab 3: Predictions
    with tabs[2]:
        st.markdown("## Recent Predictions")
        
        try:
            # Get recent predictions
            predictions_df = get_recent_predictions(limit=50)
            
            if predictions_df is not None and not predictions_df.empty:
                # Display recent predictions
                st.markdown("### Most Recent Prediction Results")
                
                # Format the dataframe for display
                display_df = predictions_df.copy()
                
                # Format the date columns
                if 'created_at' in display_df.columns:
                    display_df['created_at'] = pd.to_datetime(display_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
                
                # Format probability as percentage
                if 'probability' in display_df.columns:
                    display_df['probability'] = display_df['probability'].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A")
                
                # Show the data
                st.dataframe(display_df)
                
                # Create visualizations for predictions
                if 'prediction' in predictions_df.columns:
                    # Prediction distribution
                    st.markdown("### Prediction Distribution")
                    
                    # Count predictions by result
                    prediction_counts = predictions_df['prediction'].value_counts().reset_index()
                    prediction_counts.columns = ['Prediction', 'Count']
                    
                    # Create pie chart
                    fig = px.pie(
                        prediction_counts,
                        values='Count',
                        names='Prediction',
                        title="Distribution of Churn Predictions",
                        color='Prediction',
                        color_discrete_map={'Yes': '#e74c3c', 'No': '#2ecc71'}
                    )
                    
                    # Update traces
                    fig.update_traces(
                        textinfo='percent+label',
                        pull=[0.05, 0],
                        marker=dict(line=dict(color='white', width=2))
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Model type distribution if available
                if 'model_type' in predictions_df.columns:
                    st.markdown("### Predictions by Model Type")
                    
                    # Count predictions by model type
                    model_counts = predictions_df['model_type'].value_counts().reset_index()
                    model_counts.columns = ['Model Type', 'Count']
                    
                    # Create bar chart
                    fig2 = px.bar(
                        model_counts,
                        x='Model Type',
                        y='Count',
                        title="Predictions by Model Type",
                        color='Model Type'
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
            else:
                st.warning("No prediction data found in the database.")
                st.info("Make predictions in the 'Prediction' page to store prediction results.")
        
        except Exception as e:
            st.error(f"Error accessing prediction data: {str(e)}")
    
    # Tab 4: Advanced Database Operations
    with tabs[3]:
        st.markdown("## Advanced Database Operations")
        
        # Database status
        st.markdown("### Database Status")
        PGHOST = "localhost"
        PGPORT = "5432"
        PGUSER = "postgres"
        PGPASSWORD = "Vinny%401998"
        PGDATABASE = "customerchurn"
        DATABASE_URL = "postgresql://postgres:Vinny@1998@localhost:5432/customerchurn"
        # Check for database credentials
        db_vars = ['PGHOST', 'PGPORT', 'PGUSER', 'PGPASSWORD', 'PGDATABASE', 'DATABASE_URL']
        found_vars = [var for var in db_vars if var in os.environ]
        
        if found_vars:
            st.success(f"✅ Found database configuration: {', '.join(found_vars)}")
            
            # Display connection details (with masked password)
            if 'DATABASE_URL' in os.environ:
                db_url = os.environ.get('DATABASE_URL', '')
                # Mask password in URL for display
                if '@' in db_url and '://' in db_url:
                    parts = db_url.split('://', 1)
                    auth_host = parts[1].split('@', 1)
                    if ':' in auth_host[0]:
                        user_pass = auth_host[0].split(':', 1)
                        masked_url = f"{parts[0]}://{user_pass[0]}:******@{auth_host[1]}"
                        st.code(masked_url, language="sql")
            elif 'PGHOST' in os.environ:
                conn_string = f"postgresql://{os.environ.get('PGUSER')}:******@{os.environ.get('PGHOST')}:{os.environ.get('PGPORT')}/{os.environ.get('PGDATABASE')}"
                st.code(conn_string, language="sql")
        else:
            st.warning("⚠️ No database environment variables found. Database operations may fail.")
        
        # Test connection
        st.subheader("Connection Test")
        
        try:
            # Use direct engine creation to avoid secrets dependency
            engine = None
            if 'DATABASE_URL' in os.environ:
                engine = sa.create_engine(os.environ.get('DATABASE_URL'))
            elif all(var in os.environ for var in ['PGHOST', 'PGPORT', 'PGUSER', 'PGPASSWORD', 'PGDATABASE']):
                conn_str = f"postgresql://{os.environ['PGUSER']}:{os.environ['PGPASSWORD']}@{os.environ['PGHOST']}:{os.environ['PGPORT']}/{os.environ['PGDATABASE']}"
                engine = sa.create_engine(conn_str)
            else:
                # Try to use the engine from utils/database.py
                from utils.database import engine
            
            # Get table information
            inspector = sa.inspect(engine)
            tables = inspector.get_table_names()
            
            # Display tables and row counts
            table_data = []
            for table in tables:
                try:
                    # Count rows in table
                    with engine.connect() as conn:
                        result = conn.execute(sa.text(f"SELECT COUNT(*) FROM {table}"))
                        row_count = result.scalar()
                    
                    # Get columns
                    columns = inspector.get_columns(table)
                    column_count = len(columns)
                    
                    # Add to table data
                    table_data.append({
                        "Table Name": table,
                        "Row Count": row_count,
                        "Column Count": column_count
                    })
                except Exception as e:
                    table_data.append({
                        "Table Name": table,
                        "Row Count": f"Error: {str(e)}",
                        "Column Count": "Error"
                    })
            
            # Display table information
            if table_data:
                st.markdown("#### Database Tables")
                table_df = pd.DataFrame(table_data)
                st.dataframe(table_df)
            else:
                st.info("No tables found in the database.")
            
            # Custom SQL query section
            st.markdown("### Custom SQL Query")
            st.warning("⚠️ Use with caution: SQL operations can modify or delete data.")
            
            # SQL query input
            sql_query = st.text_area(
                "Enter a SQL query to execute:",
                height=150,
                help="Example: SELECT * FROM customers LIMIT 10"
            )
            
            # Execute query button
            if sql_query and st.button("Execute SQL Query"):
                try:
                    # Check if it's a SELECT query (safer)
                    is_select = sql_query.strip().upper().startswith("SELECT")
                    
                    with engine.connect() as conn:
                        if is_select:
                            # For SELECT queries, return and display results
                            results = pd.read_sql(sql_query, conn)
                            
                            st.success(f"Query executed successfully. {len(results)} rows returned.")
                            
                            # Show results
                            if not results.empty:
                                st.markdown("#### Query Results")
                                st.dataframe(results)
                            else:
                                st.info("Query returned no results.")
                        else:
                            # For non-SELECT queries, execute but confirm first
                            confirm = st.checkbox("I understand this query may modify the database")
                            
                            if confirm:
                                # Execute the non-SELECT query
                                result = conn.execute(sa.text(sql_query))
                                conn.commit()
                                
                                # Show result info
                                st.success(f"Query executed successfully. Rows affected: {result.rowcount}")
                            else:
                                st.warning("Please confirm to run non-SELECT queries that may modify data.")
                
                except Exception as e:
                    st.error(f"Error executing SQL query: {str(e)}")
        
        except Exception as e:
            st.error(f"Error connecting to database: {str(e)}")