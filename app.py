import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import io

def load_data():
    data_path = 'Project2/TraductionAvisClients'
    all_files = glob.glob(os.path.join(data_path, '*.xlsx'))
    dfs = []
    
    for file in all_files:
        df = pd.read_excel(file)
        dfs.append(df)
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Drop unnecessary columns
    combined_df = combined_df.drop(['avis_cor', 'avis_cor_en'], axis=1)
    
    # Remove rows with null values
    print("Shape before removing null values:", combined_df.shape)
    combined_df = combined_df.dropna()
    print("Shape after removing null values:", combined_df.shape)
    
    return combined_df

def main():
    st.title('Customer Reviews Analysis')
    
    # Load data
    with st.spinner('Loading data...'):
        df = load_data()
    
    # Display basic statistics
    st.header('Dataset Overview')
    st.write(f'Total number of reviews: {len(df)}')
    
    # Show dataset info
    st.subheader('Dataset Information')
    
    # Create two columns for better organization
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.write("**Basic Information:**")
        st.write(f"Number of rows: {df.shape[0]:,}")
        st.write(f"Number of columns: {df.shape[1]:,}")
        st.write(f"Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
    
    with info_col2:
        st.write("**Column Types:**")
        for col in df.columns:
            st.write(f"- {col}: {df[col].dtype}")
    
 
    
    # Show sample of the data
    st.subheader('Sample Reviews')
    st.dataframe(df.head())
    
    # Basic text analysis
    st.header('Text Analysis')
    
    # Calculate text statistics
    df['text_length'] = df['avis'].str.len()
    df['word_count'] = df['avis'].str.split().str.len()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('Text Length Statistics')
        st.write(df['text_length'].describe())
    
    with col2:
        st.subheader('Word Count Statistics')
        st.write(df['word_count'].describe())
    
    # Visualization
    st.header('Visualizations')

    
    # Insurance companies analysis
    st.header('Insurance Companies Analysis')
    company_counts = df['assureur'].value_counts()
    
    # Plot insurance companies distribution
    st.subheader('Distribution of Reviews by Insurance Company')
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    company_counts.head(10).plot(kind='bar', ax=ax3)
    plt.title('Top 10 Insurance Companies by Number of Reviews')
    plt.xlabel('Insurance Company')
    plt.ylabel('Number of Reviews')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig3)
    
    # Rating analysis
    st.header('Rating Analysis')
    
    # Plot rating distribution
    st.subheader('Distribution of Ratings')
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    sns.histplot(data=df, x='note', bins=5, ax=ax4)
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    st.pyplot(fig4)

if __name__ == '__main__':
    main()