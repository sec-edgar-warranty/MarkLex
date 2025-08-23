import pandas as pd
import streamlit as st
import io
import datetime
import os
from pathlib import Path
import re
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# Page config
st.set_page_config(
    page_title="MarkLex - Marketing Lexicon Creation",
    page_icon="🔍",
    layout="wide"
)

# NLTK Setup
@st.cache_resource
def setup_nltk():
    """Setup NLTK data with caching"""
    try:
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                try:
                    nltk.download('punkt_tab', quiet=True)
                except:
                    nltk.download('punkt', quiet=True)

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
            
        return set(stopwords.words('english'))
    except:
        return {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}

# Load data with caching
@st.cache_data
def load_lexicon():
    """Load lexicon data with caching"""
    try:
        lexicon_path = "Lexicon List.xlsx"
        if os.path.exists(lexicon_path):
            df = pd.read_excel(lexicon_path)
            return df
        else:
            return create_default_lexicon()
    except Exception as e:
        st.error(f"Error loading lexicon: {str(e)}")
        return create_default_lexicon()

def create_default_lexicon():
    """Create a default lexicon"""
    return pd.DataFrame({
        'Entity': ['Marketing'] * 4 + ['ESG'] * 4 + ['Risk'] * 4,
        'Keyword': ['marketing', 'advertising', 'promotion', 'brand',
                   'sustainability', 'environmental', 'governance', 'social',
                   'risk', 'security', 'compliance', 'audit']
    })

@st.cache_resource
def load_embedding_model(model_type='uni'):
    """Load embedding model with caching"""
    candidates = ['embeddings_8', 'embeddings_bi_grams'] if model_type == 'uni' else ['embeddings_bi_grams', 'embeddings_8']
    
    for candidate in candidates:
        if os.path.exists(candidate):
            try:
                with st.spinner(f"Loading {model_type} model... This may take a moment on first run."):
                    model = Word2Vec.load(candidate)
                return model
            except Exception as e:
                st.error(f"Failed to load model: {e}")
    
    st.error(f"❌ No {model_type} embedding model found!")
    return None

def create_lexicon_from_word(seed_word, num_words, model_type='uni'):
    """Create lexicon from seed word"""
    try:
        model = load_embedding_model(model_type)
        if model is None:
            return pd.DataFrame({'Error': ['Embedding model not found']})

        seed_word_formatted = seed_word.lower().replace(' ', '_')

        if seed_word_formatted not in model.wv.key_to_index:
            return pd.DataFrame({'Error': [f'Word "{seed_word}" not found in vocabulary']})

        sims = model.wv.most_similar(seed_word_formatted, topn=num_words)
        df = pd.DataFrame(sims, columns=['Similar Word', 'Similarity Score'])
        df['Similarity Score'] = df['Similarity Score'].round(3)
        df.insert(0, 'Word Number', range(1, len(df) + 1))

        return df

    except Exception as e:
        return pd.DataFrame({'Error': [f'Error: {str(e)}']})

# Text processing functions
def preprocess_text(text):
    text = text.replace("U.S.A.", "USA").replace("U.S.", "US")
    return text

def tokenize_sentences(text, stop_words):
    try:
        sentences = sent_tokenize(preprocess_text(text))
        return [s for s in sentences if s.strip()]
    except:
        sentences = re.split(r'[.!?]+', preprocess_text(text))
        return [s.strip() for s in sentences if s.strip()]

def clean_text(sentences, stop_words):
    cleaned_sentences = []
    for i, sentence in enumerate(sentences):
        try:
            text = sentence.lower()
            text = re.sub(r'[^\w\s]', '', text)
            words = text.split()
            words = [word for word in words if word and word not in stop_words and len(word) > 1]

            if words:
                cleaned_sentences.append({
                    'sentence_id': i + 1,
                    'text': ' '.join(words),
                    'original': sentence
                })
        except:
            continue

    return cleaned_sentences

def generate_ngrams(cleaned_sentences, n):
    ngrams = []
    for sentence in cleaned_sentences:
        words = sentence['text'].split()
        if len(words) >= n:
            for i in range(len(words) - n + 1):
                ngram = ' '.join(words[i:i + n])
                if ngram.strip():
                    ngrams.append({
                        'sentence_id': sentence['sentence_id'],
                        'ngram': ngram
                    })
    return ngrams

def analyze_text(text_input, lexicon, stop_words):
    if not text_input or not text_input.strip():
        return pd.DataFrame({'Text': ['No text provided']})

    sentences = tokenize_sentences(text_input, stop_words)
    if not sentences:
        return pd.DataFrame({'Text': ['No sentences found']})

    cleaned_sentences = clean_text(sentences, stop_words)
    if not cleaned_sentences:
        return pd.DataFrame({'Text': sentences})

    # Generate n-grams
    all_ngrams = generate_ngrams(cleaned_sentences, 1) + generate_ngrams(cleaned_sentences, 2) + generate_ngrams(cleaned_sentences, 3)

    if all_ngrams:
        ngrams_df = pd.DataFrame(all_ngrams)
    else:
        ngrams_df = pd.DataFrame(columns=['sentence_id', 'ngram'])

    results = {'Text': sentences}

    if lexicon.empty:
        return pd.DataFrame(results)

    # Process each entity
    for entity in lexicon['Entity'].unique():
        keywords = lexicon[lexicon['Entity'] == entity]['Keyword'].tolist()
        keywords = [str(k).lower() for k in keywords if pd.notna(k)]

        entity_matches = []

        for i, sentence in enumerate(sentences):
            sentence_id = i + 1

            if not ngrams_df.empty:
                sentence_ngrams = ngrams_df[ngrams_df['sentence_id'] == sentence_id]['ngram'].tolist()
            else:
                sentence_ngrams = []

            match = 1 if any(ngram in keywords for ngram in sentence_ngrams) else 0
            entity_matches.append(match)

        results[entity] = entity_matches

    return pd.DataFrame(results)

@st.cache_data
def read_markdown_file(filename):
    try:
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as file:
                return file.read()
    except:
        pass

    if 'welcome' in filename.lower():
        return """# Welcome to MarkLex

**MarkLex - Marketing Lexicon Creation**

This application helps you create marketing lexicons and analyze text for specific themes and keywords.

## Features

- **Lexicon Creation**: Generate word lists based on seed words using pre-trained embeddings
- **Text Analysis**: Analyze text documents for various business dimensions
- **Export Data**: Download your results as CSV files

Navigate through the tabs to explore the different features."""
    else:
        return """# Instructions

## Lexicon Tab
Use unigrams (single words) or bigrams (two words) to generate related terms based on semantic similarity.

## Text Analysis Tab
Analyze text for predefined business dimensions including Marketing, ESG, Risk, and more."""

@st.fragment
def lexicon_tab_content():
    """Lexicon tab content wrapped in fragment to prevent tab switching"""
    st.header("Lexicon Creation")
    
    # Initialize session state for lexicon data
    if 'lexicon_data' not in st.session_state:
        st.session_state.lexicon_data = None
    if 'lexicon_term' not in st.session_state:
        st.session_state.lexicon_term = ""
    if 'lexicon_input_term' not in st.session_state:
        st.session_state.lexicon_input_term = ""
    if 'lexicon_n_words' not in st.session_state:
        st.session_state.lexicon_n_words = 20
    
    # Quick instructions at the top
    st.info("💡 **Quick Start**: Enter a unigram (e.g., 'marketing') or bigram (e.g., 'profit margin') below, set the number of words, and click 'Create Lexicon' to generate semantically similar terms.")
    
    # Create columns for inputs
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Inputs
        term = st.text_input(
            "Add a term (max 2 words):",
            value=st.session_state.lexicon_input_term,
            placeholder="sustainability",
            help="Enter a single word or two words separated by a space",
            key="lexicon_term_input"
        )
        
        n_words = st.slider(
            "Number of words:",
            min_value=10,
            max_value=100,
            value=st.session_state.lexicon_n_words,
            step=5,
            key="lexicon_n_words_slider"
        )
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            run_lexicon = st.button("Create Lexicon", type="primary")
        with col_btn2:
            reset_lexicon = st.button("Reset", type="secondary")
    
    with col2:
        # Handle reset
        if reset_lexicon:
            st.session_state.lexicon_data = None
            st.session_state.lexicon_term = ""
            st.session_state.lexicon_input_term = ""
            st.session_state.lexicon_n_words = 20
            st.success("Lexicon data cleared!")
            st.rerun()
        
        # Update session state with current inputs
        st.session_state.lexicon_input_term = term
        st.session_state.lexicon_n_words = n_words
        
        # Handle lexicon creation
        if run_lexicon and term:
            num_words = len(term.split())
            model_type = 'uni' if num_words == 1 else 'bi'
            
            with st.spinner("Creating lexicon..."):
                result = create_lexicon_from_word(term, n_words, model_type)
            
            if 'Error' in result.columns:
                st.error(result['Error'].iloc[0])
                st.session_state.lexicon_data = None
            else:
                # Store in session state
                st.session_state.lexicon_data = result
                st.session_state.lexicon_term = term
        
        elif run_lexicon:
            st.warning("Please enter a term to create a lexicon.")
        
        # Display lexicon data (persistent)
        if st.session_state.lexicon_data is not None:
            st.dataframe(st.session_state.lexicon_data, use_container_width=True)
            
            # Download and Reset buttons side by side
            col_dl, col_reset = st.columns([2, 1])
            with col_dl:
                csv = st.session_state.lexicon_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Lexicon",
                    data=csv,
                    file_name=f"lexicon_{st.session_state.lexicon_term}_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        else:
            st.info("The lexicon will appear here.")
    
    # Detailed instructions - collapsed by default
    st.subheader("📚 Detailed Instructions")
    with st.expander("6-Step Lexicon Development Process", expanded=False):
        st.markdown("""
        **Follow this systematic approach for building high-quality lexicons:**
        
        **Step 1: Enter Construct**
        - Enter a unigram (single term: marketing, profit, privacy) or bigram (two terms: profit margin, customer satisfaction)
        - Use domain-specific terminology relevant to your research construct
        
        **Step 2: Export Top-N Neighbors**
        - Start with N=50 similar words to get initial coverage
        - Expand to N=100 if recall seems thin or you need broader coverage
        - Download the results as CSV for systematic review
        
        **Step 3: Iterative Expansion**
        - Promote good neighbors to seeds for the next iteration
        - Re-run the process 1–2 times to widen coverage
        - Domain-specific variants emerge quickly through this iterative approach
        
        **Step 4: Prune & Disambiguate**
        - Remove irrelevant or ambiguous terms from your lexicon
        - Prefer bigrams where needed to reduce ambiguity
        - Focus on terms that clearly represent your target construct
        
        **Step 5: Validation**
        - Conduct human review with Inter-Annotator Agreement (IAA) on a sentence sample
        - Report key metrics: Accuracy, Precision, Recall, and F1 score
        - Document validation methodology and results
        
        **Step 6: Documentation & Versioning**
        - Version your final lexicon (e.g., v1.0, v1.1)
        - Include provenance notes: seed words used, app version, source corpus
        - Maintain documentation for reproducibility and future updates
        
        **Troubleshooting**
        
        If the app fails to return results, try variations of your search terms or related concepts.
        Not all terms may be present in the vocabulary - experiment with synonyms or broader/narrower terms.
        """)

def main():
    # Initialize data
    stop_words = setup_nltk()
    lexicon = load_lexicon()
    
    # Load markdown content
    welcome_content = read_markdown_file("welcome.md")
    instructions_content = read_markdown_file("instructions.md")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["MarkLex", "Lexicon", "Text Analysis"])
    
    with tab1:
        st.markdown(welcome_content)
    
    with tab2:
        lexicon_tab_content()  # Call the fragment function
    
    with tab3:
        st.header("Text Analysis")
        
        # Initialize session state for text analysis data
        if 'analysis_data' not in st.session_state:
            st.session_state.analysis_data = None
        if 'analysis_text_input' not in st.session_state:
            st.session_state.analysis_text_input = ""
        if 'analysis_input_text' not in st.session_state:
            st.session_state.analysis_input_text = ""
        
        # Instructions
        st.subheader("Instructions")
        st.info("""
        This section provides a demonstration of the use of the lexicon. Here, we have created a lexicon 
        that comprises eight dimensions: Marketing, ESG, DEI, Risk & Security, Employee, AI & Data, 
        Innovation, and Investor Focus. Paste your text in the space provided below and click Analyze Text. 
        The output table will show you each sentence and the number of words associated with these dimensions. 
        You can also output this table as a spreadsheet for further analysis.
        """)
        
        # Text input
        text_input = st.text_area(
            "Input text:",
            value=st.session_state.analysis_input_text,
            height=150,
            placeholder="Paste your text here for analysis...",
            key="analysis_text_area"
        )
        
        # Buttons
        col1, col2, col3 = st.columns([0.1, 0.1, 0.8])
        with col1:
            analyze_button = st.button("Analyze Text", type="primary")
        with col2:
            reset_analysis = st.button("Reset Analysis", type="secondary")
        
        # Handle reset
        if reset_analysis:
            st.session_state.analysis_data = None
            st.session_state.analysis_text_input = ""
            st.session_state.analysis_input_text = ""
            st.success("Analysis data cleared!")
            st.rerun()
        
        # Update session state with current input
        st.session_state.analysis_input_text = text_input
        
        # Handle analysis
        if analyze_button and text_input:
            with st.spinner("Analyzing text..."):
                result = analyze_text(text_input, lexicon, stop_words)
            
            # Store in session state
            st.session_state.analysis_data = result
            st.session_state.analysis_text_input = text_input[:100] + "..." if len(text_input) > 100 else text_input
            
        elif analyze_button:
            st.warning("Please enter some text to analyze.")
        
        # Display analysis results (persistent)
        if st.session_state.analysis_data is not None:
            st.subheader("Analysis Results")
            st.dataframe(st.session_state.analysis_data, use_container_width=True)
            
            # Download button
            col_dl, col_space = st.columns([2, 2])
            with col_dl:
                csv = st.session_state.analysis_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Analysis Data",
                    data=csv,
                    file_name=f"text_analysis_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        else:
            st.info("Analysis results will appear here.")

if __name__ == "__main__":
    main()
    
    # Fix for tab switching issue - force initial rerun to stabilize state
    if "initial_rerun_done" not in st.session_state:
        st.session_state.initial_rerun_done = True
        st.rerun()