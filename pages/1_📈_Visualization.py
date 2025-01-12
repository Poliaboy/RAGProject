import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from collections import Counter
import re
from nltk.util import ngrams
import nltk
from nltk.corpus import stopwords
import spacy

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Load spaCy models
try:
    nlp_en = spacy.load('en_core_web_sm')
    nlp_fr = spacy.load('fr_core_news_sm')
except OSError:
    st.warning("Downloading spaCy models...")
    os.system('python -m spacy download en_core_web_sm')
    os.system('python -m spacy download fr_core_news_sm')
    nlp_en = spacy.load('en_core_web_sm')
    nlp_fr = spacy.load('fr_core_news_sm')

st.set_page_config(
    page_title="Data Cleaning & Analysis",
    page_icon="📈",
    layout="wide"
)

def remove_stopwords(text, lang='en'):
    """Remove stopwords from text using spaCy"""
    if pd.isna(text) or text == "":
        return ""
    
    # Use spaCy for stopwords
    nlp = nlp_en if lang == 'en' else nlp_fr
    doc = nlp(text)
    return ' '.join([token.text for token in doc if not token.is_stop])

def lemmatize_text(text, lang='en'):
    """Lemmatize text using spaCy"""
    if pd.isna(text) or text == "":
        return ""
    
    nlp = nlp_en if lang == 'en' else nlp_fr
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])

def load_and_clean_data():
    """Load and clean the data"""
    # Check if processed data exists
    processed_data_path = 'data/processed_data.pkl'
    if os.path.exists(processed_data_path):
        st.write("Loading pre-processed data...")
        return pd.read_pickle(processed_data_path)
    
    st.write("Processing data for the first time...")
    data_path = 'TraductionAvisClients'
    all_files = glob.glob(os.path.join(data_path, '*.xlsx'))
    dfs = []
    
    for file in all_files:
        df = pd.read_excel(file)
        dfs.append(df)
    
    # Combine all dataframes
    df = pd.concat(dfs, ignore_index=True)
    
    # Display initial data stats
    st.subheader("Initial Data Statistics")
    st.write(f"Initial shape: {df.shape}")
    
    # Remove unnecessary columns
    columns_to_drop = ['avis_cor', 'avis_cor_en']
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    # Handle missing values
    missing_stats_before = df.isnull().sum()
    st.write("Missing values before cleaning:")
    st.write(missing_stats_before[missing_stats_before > 0])
    
    # Remove rows with missing values in critical columns
    critical_columns = ['note', 'avis', 'avis_en', 'assureur', 'produit']
    df = df.dropna(subset=critical_columns)
    
    with st.spinner("Cleaning and processing text..."):
        # Basic cleaning
        df['avis_cleaned'] = df['avis'].apply(clean_text)
        df['avis_en_cleaned'] = df['avis_en'].apply(clean_text)
        
        # Process French text
        st.write("Processing French text...")
        df['avis_no_stop'] = df['avis_cleaned'].apply(lambda x: remove_stopwords(x, 'fr'))
        df['avis_lemmatized'] = df['avis_cleaned'].apply(lambda x: lemmatize_text(x, 'fr'))
        df['avis_no_stop_and_lemmatized'] = df['avis_cleaned'].apply(lambda x: lemmatize_text(remove_stopwords(x, 'fr'), 'fr'))
        
        # Process English text
        st.write("Processing English text...")
        df['avis_en_no_stop'] = df['avis_en_cleaned'].apply(lambda x: remove_stopwords(x, 'en'))
        df['avis_en_lemmatized'] = df['avis_en_cleaned'].apply(lambda x: lemmatize_text(x, 'en'))
        df['avis_en_no_stop_and_lemmatized'] = df['avis_en_cleaned'].apply(lambda x: lemmatize_text(remove_stopwords(x, 'en'), 'en'))
    # Add text statistics
    text_columns = ['avis', 'avis_en', 'avis_cleaned', 'avis_en_cleaned', 
                   'avis_no_stop', 'avis_en_no_stop', 
                   'avis_lemmatized', 'avis_en_lemmatized']
    
    for col in text_columns:
        df[f'{col}_length'] = df[col].str.len()
        df[f'{col}_word_count'] = df[col].str.split().str.len()
    
    # Save processed data
    os.makedirs('data', exist_ok=True)
    df.to_pickle(processed_data_path)
    st.success("Data processed and saved!")
    
    return df

def clean_text(text):
    """Enhanced text cleaning function"""
    if pd.isna(text):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def get_ngrams(text, n, lang='en'):
    """Extract n-grams from text using spaCy"""
    if pd.isna(text) or text == "":
        return []
    
    # Use spaCy for tokenization
    nlp = nlp_en if lang == 'en' else nlp_fr
    doc = nlp(text)
    
    # Get tokens, excluding stopwords and punctuation
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
    
    # Generate n-grams
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(' '.join(tokens[i:i+n]))
    
    return ngrams

st.title('📈 Data Cleaning & Analysis')

# Add explanation of what we're doing
st.markdown("""
### What are we analyzing?
This section provides comprehensive analysis of customer reviews data through multiple lenses:

1. **Data Quality Analysis**: 
   - Text cleaning and preprocessing
   - Handling missing values and duplicates
   - Comparing original and processed text versions
   - Examining data distributions

2. **Text Statistics**: 
   - Review length analysis
   - Word count distributions
   - Language comparison (French vs English)
   - Statistical summaries

3. **N-gram Analysis**:
   - Discover common phrases and patterns
   - Compare language-specific patterns
   - Identify frequent word combinations
   - Analyze contextual usage

4. **Company Analysis**:
   - Company-specific review patterns
   - Rating distributions
   - Product breakdowns
   - Review length comparisons

### Why is this important?
- **Data Quality**: Ensures our models work with clean, consistent data
- **Text Insights**: Reveals patterns in customer feedback
- **Business Intelligence**: Provides actionable insights for companies
- **Preprocessing Validation**: Confirms our text processing is effective
""")

# Load and clean data
with st.spinner('Loading and cleaning data...'):
    df = load_and_clean_data()

# Create tabs for different analyses
tab1, tab2, tab3, tab4 = st.tabs([
    "Data Quality",
    "Text Statistics",
    "N-gram Analysis",
    "Company Analysis"
])

with tab1:
    st.header('Data Quality Analysis')
    st.markdown("""
    This tab shows how we clean and prepare the text data. Compare original and processed versions to understand:
    - Basic text cleaning (lowercase, special characters)
    - Stopword removal impact
    - Lemmatization effects
    - Data quality metrics
    """)
    
    # Display data types
    st.subheader("Data Types")
    st.write(df.dtypes)
    
    # Display sample data
    st.subheader("Sample Data (After Cleaning)")
    st.dataframe(df.head())
    
    # Text cleaning example
    st.subheader("Text Processing Examples")
    sample_idx = np.random.randint(len(df))
    
    # French examples
    st.markdown("### French Text Processing")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original Text:**")
        st.write(df['avis'].iloc[sample_idx])
        st.markdown("**Cleaned Text:**")
        st.write(df['avis_cleaned'].iloc[sample_idx])
    with col2:
        st.markdown("**Without Stopwords:**")
        st.write(df['avis_no_stop'].iloc[sample_idx])
        st.markdown("**Lemmatized:**")
        st.write(df['avis_lemmatized'].iloc[sample_idx])
        st.markdown("**Without Stopwords and Lemmatized:**")
        st.write(df['avis_no_stop_and_lemmatized'].iloc[sample_idx])
    
    # English examples
    st.markdown("### English Text Processing")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original Text:**")
        st.write(df['avis_en'].iloc[sample_idx])
        st.markdown("**Cleaned Text:**")
        st.write(df['avis_en_cleaned'].iloc[sample_idx])
    with col2:
        st.markdown("**Without Stopwords:**")
        st.write(df['avis_en_no_stop'].iloc[sample_idx])
        st.markdown("**Lemmatized:**")
        st.write(df['avis_en_lemmatized'].iloc[sample_idx])
        st.markdown("**Without Stopwords and Lemmatized:**")
        st.write(df['avis_en_no_stop_and_lemmatized'].iloc[sample_idx])
    
    # Text statistics comparison
    st.subheader("Text Processing Statistics")
    stats_comparison = pd.DataFrame({
        'Metric': ['Average Word Count', 'Average Length'],
        'Original (FR)': [df['avis_word_count'].mean(), df['avis_length'].mean()],
        'Cleaned (FR)': [df['avis_cleaned_word_count'].mean(), df['avis_cleaned_length'].mean()],
        'No Stopwords (FR)': [df['avis_no_stop_word_count'].mean(), df['avis_no_stop_length'].mean()],
        'Lemmatized (FR)': [df['avis_lemmatized_word_count'].mean(), df['avis_lemmatized_length'].mean()],
        'Original (EN)': [df['avis_en_word_count'].mean(), df['avis_en_length'].mean()],
        'Cleaned (EN)': [df['avis_en_cleaned_word_count'].mean(), df['avis_en_cleaned_length'].mean()],
        'No Stopwords (EN)': [df['avis_en_no_stop_word_count'].mean(), df['avis_en_no_stop_length'].mean()],
        'Lemmatized (EN)': [df['avis_en_lemmatized_word_count'].mean(), df['avis_en_lemmatized_length'].mean()]
    })
    st.dataframe(stats_comparison.round(2))
    
    # Display duplicate statistics
    st.subheader("Duplicate Analysis")
    duplicates = pd.DataFrame({
        'Text Version': ['Original (FR)', 'Cleaned (FR)', 'No Stopwords (FR)', 'Lemmatized (FR)',
                        'Original (EN)', 'Cleaned (EN)', 'No Stopwords (EN)', 'Lemmatized (EN)'],
        'Duplicate Count': [
            df['avis'].duplicated().sum(),
            df['avis_cleaned'].duplicated().sum(),
            df['avis_no_stop'].duplicated().sum(),
            df['avis_lemmatized'].duplicated().sum(),
            df['avis_en'].duplicated().sum(),
            df['avis_en_cleaned'].duplicated().sum(),
            df['avis_en_no_stop'].duplicated().sum(),
            df['avis_en_lemmatized'].duplicated().sum()
        ]
    })
    st.dataframe(duplicates)

with tab2:
    st.header('Text Statistics')
    st.markdown("""
    Analyze the statistical properties of reviews:
    - Review length patterns
    - Word count distributions
    - Language comparisons
    - Identify outliers and trends
    """)
    
    # Language selection
    language = st.radio("Select Language", ["French", "English"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('Text Length Distribution')
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        # Fix column names
        length_col = 'avis_length' if language == "French" else 'avis_en_length'
        sns.histplot(data=df, x=length_col, bins=50)
        plt.title(f'Distribution of Review Text Length ({language})')
        plt.xlabel('Text Length (characters)')
        plt.ylabel('Count')
        st.pyplot(fig1)
    
    with col2:
        st.subheader('Word Count Distribution')
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        # Fix column names
        count_col = 'avis_word_count' if language == "French" else 'avis_en_word_count'
        sns.histplot(data=df, x=count_col, bins=50)
        plt.title(f'Distribution of Review Word Count ({language})')
        plt.xlabel('Word Count')
        plt.ylabel('Count')
        st.pyplot(fig2)
    
    # Display summary statistics
    st.subheader("Text Statistics Summary")
    stats_df = pd.DataFrame({
        'Metric': ['Mean Length', 'Max Length', 'Mean Word Count', 'Max Word Count'],
        'French': [
            df['avis_length'].mean(),
            df['avis_length'].max(),
            df['avis_word_count'].mean(),
            df['avis_word_count'].max()
        ],
        'English': [
            df['avis_en_length'].mean(),
            df['avis_en_length'].max(),
            df['avis_en_word_count'].mean(),
            df['avis_en_word_count'].max()
        ]
    })
    st.dataframe(stats_df.round(2))

with tab3:
    st.header('N-gram Analysis')
    st.markdown("""
    Discover common phrases and patterns in reviews:
    - Most frequent word combinations
    - Language-specific patterns
    - Common expressions
    - Context analysis
    """)
    
    # Language and n-gram selection
    col1, col2 = st.columns(2)
    with col1:
        language = st.radio("Select Language for N-grams", ["French", "English"], key="ngram_lang")
    with col2:
        n = st.selectbox("Select n-gram size:", [1, 2, 3])
    
    if st.button(f"Analyze {n}-grams"):
        with st.spinner(f"Analyzing {n}-grams..."):
            # Get sample of reviews for n-gram analysis
            text_col = 'avis_cleaned' if language == "French" else 'avis_en_cleaned'
            sample_reviews = df[text_col].head(1000)
            
            # Extract n-grams
            all_ngrams = []
            lang = 'fr' if language == "French" else 'en'
            for review in sample_reviews:
                all_ngrams.extend(get_ngrams(review, n, lang))
            
            # Count n-grams
            ngram_counts = Counter(all_ngrams)
            
            # Display results
            st.subheader(f"Most Common {n}-grams ({language})")
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(12, 6))
            top_ngrams = dict(sorted(ngram_counts.items(), key=lambda x: x[1], reverse=True)[:20])
            
            if top_ngrams:  # Check if we have any n-grams
                plt.bar(top_ngrams.keys(), top_ngrams.values())
                plt.xticks(rotation=45, ha='right')
                plt.title(f'Top 20 {n}-grams')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Display table
                ngram_df = pd.DataFrame(list(top_ngrams.items()), columns=['N-gram', 'Frequency'])
                st.dataframe(ngram_df)
            else:
                st.warning("No n-grams found in the sample text.")

with tab4:
    st.header('Company Analysis')
    st.markdown("""
    Deep dive into company-specific patterns:
    - Rating distributions
    - Product popularity
    - Review characteristics
    - Company comparisons
    """)
    
    # Company selection
    selected_company = st.selectbox(
        "Select Insurance Company",
        options=sorted(df['assureur'].unique())
    )
    
    if selected_company:
        company_data = df[df['assureur'] == selected_company]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Rating distribution
            st.subheader('Rating Distribution')
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            sns.histplot(data=company_data, x='note', bins=5)
            plt.title(f'Rating Distribution for {selected_company}')
            plt.xlabel('Rating')
            plt.ylabel('Count')
            st.pyplot(fig3)
        
        with col2:
            # Product distribution
            st.subheader('Product Distribution')
            product_counts = company_data['produit'].value_counts()
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            plt.pie(product_counts.values, labels=product_counts.index, autopct='%1.1f%%')
            plt.title(f'Product Distribution for {selected_company}')
            st.pyplot(fig4)
        
        # Show detailed statistics
        st.subheader('Company Statistics')
        stats = {
            'Total Reviews': len(company_data),
            'Average Rating': company_data['note'].mean(),
            'Most Common Product': company_data['produit'].mode()[0],
            'Average Review Length (FR)': int(company_data['avis_length'].mean()),
            'Average Review Length (EN)': int(company_data['avis_en_length'].mean()),
            'Average Word Count (FR)': int(company_data['avis_word_count'].mean()),
            'Average Word Count (EN)': int(company_data['avis_en_word_count'].mean())
        }
        
        stats_df = pd.DataFrame([stats])
        st.dataframe(stats_df) 
