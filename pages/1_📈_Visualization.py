import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

st.set_page_config(
    page_title="Data Visualization",
    page_icon="📈",
    layout="wide"
)

def load_data():
    data_path = 'TraductionAvisClients'
    all_files = glob.glob(os.path.join(data_path, '*.xlsx'))
    dfs = []
    
    for file in all_files:
        df = pd.read_excel(file)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.drop(['avis_cor', 'avis_cor_en'], axis=1)
    
    # Calculate text statistics
    combined_df['text_length'] = combined_df['avis'].str.len()
    combined_df['word_count'] = combined_df['avis'].str.split().str.len()
    
    return combined_df

st.title('📈 Data Visualization')

# Load data
with st.spinner('Loading data...'):
    df = load_data()

# Create tabs for different visualizations
tab1, tab2, tab3, tab4 = st.tabs([
    "Text Analysis", 
    "Insurance Companies", 
    "Product Analysis",
    "Rating Distribution"
])

with tab1:
    st.header('Text Analysis')
    
    # Text statistics in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('Text Length Statistics')
        # Convert describe() output to dataframe and display with custom width
        text_stats = pd.DataFrame(df['text_length'].describe())
        st.dataframe(text_stats, width=300)
        
        # Text length distribution
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.histplot(data=df, x='text_length', bins=50, ax=ax1)
        plt.title('Distribution of Review Text Length')
        plt.xlabel('Text Length (characters)')
        plt.ylabel('Count')
        st.pyplot(fig1)
    
    with col2:
        st.subheader('Word Count Statistics')
        # Convert describe() output to dataframe and display with custom width
        word_stats = pd.DataFrame(df['word_count'].describe())
        st.dataframe(word_stats, width=300)
        
        # Word count distribution
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.histplot(data=df, x='word_count', bins=50, ax=ax2)
        plt.title('Distribution of Review Word Count')
        plt.xlabel('Word Count')
        plt.ylabel('Count')
        st.pyplot(fig2)

with tab2:
    st.header('Insurance Companies Analysis')
    
    # Insurance companies distribution
    company_counts = df['assureur'].value_counts()
    
    # Plot top insurance companies
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    company_counts.head(10).plot(kind='bar', ax=ax3)
    plt.title('Top 10 Insurance Companies by Number of Reviews')
    plt.xlabel('Insurance Company')
    plt.ylabel('Number of Reviews')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig3)
    
    # Show full company distribution in a table
    st.subheader('Complete Insurance Company Distribution')
    company_dist_df = pd.DataFrame({
        'Company': company_counts.index,
        'Number of Reviews': company_counts.values,
        'Percentage': (company_counts.values / len(df) * 100).round(2)
    })
    st.dataframe(company_dist_df)

with tab3:
    st.header('Product Analysis')
    
    # Product type distribution
    product_counts = df['produit'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        product_counts.plot(kind='bar', ax=ax4)
        plt.title('Distribution of Product Types')
        plt.xlabel('Product Type')
        plt.ylabel('Number of Reviews')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig4)
    
    with col2:
        # Pie chart
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        plt.pie(product_counts, labels=product_counts.index, autopct='%1.1f%%')
        plt.title('Product Type Distribution (Percentage)')
        plt.axis('equal')
        st.pyplot(fig5)

with tab4:
    st.header('Rating Analysis')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Rating distribution
        fig6, ax6 = plt.subplots(figsize=(10, 6))
        sns.histplot(data=df, x='note', bins=5, ax=ax6)
        plt.title('Distribution of Ratings')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        st.pyplot(fig6)
        # Rating statistics
        st.subheader('Rating Statistics')
        rating_stats = pd.DataFrame(df['note'].describe())
        st.dataframe(rating_stats, width=300)
        
    
    with col2:
        
        # Average rating by product type
        avg_rating_by_product = df.groupby('produit')['note'].mean().sort_values(ascending=False)
        st.subheader('Average Rating by Product Type')
        st.dataframe(avg_rating_by_product, width=400) 
