import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import io

st.set_page_config(
    page_title="Customer Reviews Analysis",
    page_icon="📊",
    layout="wide"
)

def load_data():
    data_path = 'TraductionAvisClients'
    all_files = glob.glob(os.path.join(data_path, '*.xlsx'))
    dfs = []
    
    for file in all_files:
        df = pd.read_excel(file)
        dfs.append(df)
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Drop unnecessary columns
    combined_df = combined_df.drop(['avis_cor', 'avis_cor_en'], axis=1)
    
    return combined_df

def main():
    st.title('📊 Customer Reviews Analysis Dashboard')
    
    st.markdown("""
                
    ## Welcome to the Customer Reviews Analysis Dashboard
    
    This application provides comprehensive analysis of customer reviews for various insurance products and companies.
    
    ### Available Features:
    
    🔍 **Data Overview**
    - Basic statistics and dataset information
    - Sample reviews visualization
    
    📈 **Data Visualization** (See Visualization page)
    - Text length and word count analysis
    - Insurance companies distribution
    - Rating distribution
    - Product type analysis
    
    💬 **Chat Interface** (See Chat page)
    - Interactive chat interface for data queries
    - Natural language processing capabilities
    
    ### Getting Started
    
    Use the sidebar navigation to explore different sections of the dashboard. Each page provides unique insights into the customer reviews data.
    """)
    
    # Load data
    with st.spinner('Loading data...'):
        df = load_data()
    
    # Display key metrics in a nice format using columns
    st.header('Quick Overview')
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reviews", f"{len(df):,}")
    
    with col2:
        st.metric("Insurance Companies", f"{df['assureur'].nunique():,}")
    
    with col3:
        st.metric("Product Types", f"{df['produit'].nunique():,}")
    
    with col4:
        st.metric("Average Review Length", f"{int(df['avis'].str.len().mean()):,} chars")

if __name__ == '__main__':
    main()