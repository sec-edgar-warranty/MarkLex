import pandas as pd
from pathlib import Path
from shiny import App, Inputs, Outputs, Session, reactive, render, ui
import asyncio
import re
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import io
import datetime
import sys
import os

# NLTK Setup for standalone app
def setup_nltk():
    """Setup NLTK data with fallbacks for standalone app"""
    try:
        try:
            nltk.data.find('tokenizers/punkt_tab')
            print("Found punkt_tab")
        except LookupError:
            try:
                nltk.data.find('tokenizers/punkt')
                print("Found punkt")
            except LookupError:
                try:
                    nltk.download('punkt_tab', quiet=True)
                    print("Downloaded punkt_tab")
                except:
                    pass
                try:
                    nltk.download('punkt', quiet=True)
                    print("Downloaded punkt")
                except:
                    pass

        try:
            nltk.data.find('corpora/stopwords')
            print("Found stopwords")
        except LookupError:
            nltk.download('stopwords', quiet=True)
            print("Downloaded stopwords")

    except Exception as e:
        print(f"NLTK setup warning: {e}")

setup_nltk()

# Enhanced resource path resolver
def resource_path(relative_path: str) -> str:
    """Return an absolute path to a resource, works for dev and PyInstaller bundles."""
    base_paths = []

    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
        base_paths.append(base_path)
        print(f"Found PyInstaller temp path: {base_path}")
    except AttributeError:
        # Not running in PyInstaller bundle
        pass

    # Add current script directory
    script_dir = Path(__file__).parent
    base_paths.append(str(script_dir))

    # Add current working directory
    base_paths.append(os.getcwd())

    # Add some common resource locations
    if hasattr(sys, 'executable'):
        exec_dir = Path(sys.executable).parent
        base_paths.extend([
            str(exec_dir),
            str(exec_dir / 'Resources'),
            str(exec_dir / '..' / 'Resources'),
        ])

    # Try each base path
    for base_path in base_paths:
        full_path = os.path.join(str(base_path), relative_path)
        if os.path.exists(full_path):
            print(f"✓ Found resource: {full_path}")
            return full_path

    # If not found, return the first attempted path for error reporting
    fallback_path = os.path.join(str(base_paths[0]) if base_paths else os.getcwd(), relative_path)
    print(f"⚠ Resource not found anywhere, using fallback: {fallback_path}")
    return fallback_path

# Load functions
def load_lexicon():
    """Load lexicon data with comprehensive error handling"""
    try:
        lexicon_path = resource_path("Lexicon List.xlsx")
        print(f"Loading lexicon from: {lexicon_path}")

        if not os.path.exists(lexicon_path):
            print(f"❌ Lexicon file does not exist: {lexicon_path}")
            return create_default_lexicon()

        df = pd.read_excel(lexicon_path)
        print(f"✓ Loaded lexicon with {len(df)} rows")
        return df

    except Exception as e:
        print(f"❌ Error loading lexicon: {str(e)}")
        return create_default_lexicon()

def create_default_lexicon():
    """Create a default lexicon"""
    print("Creating default lexicon...")
    return pd.DataFrame({
        'Entity': ['Marketing'] * 4 + ['ESG'] * 4 + ['Risk'] * 4,
        'Keyword': ['marketing', 'advertising', 'promotion', 'brand',
                   'sustainability', 'environmental', 'governance', 'social',
                   'risk', 'security', 'compliance', 'audit']
    })

def load_embedding_model(model_type='uni'):
    """Load embedding model"""
    candidates = ['embeddings_8', 'embeddings_bi_grams'] if model_type == 'uni' else ['embeddings_bi_grams', 'embeddings_8']

    for candidate in candidates:
        model_path = resource_path(candidate)
        if os.path.exists(model_path):
            try:
                model = Word2Vec.load(model_path)
                print(f"✓ Loaded {model_type} model")
                return model
            except Exception as e:
                print(f"✗ Failed to load model: {e}")

    print(f"❌ No {model_type} embedding model found!")
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

# Initialize components
lexicon = load_lexicon()
try:
    stop_words = set(stopwords.words('english'))
except:
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}

# Text processing functions
def preprocess_text(text):
    text = text.replace("U.S.A.", "USA").replace("U.S.", "US")
    return text

def tokenize_sentences(text):
    try:
        sentences = sent_tokenize(preprocess_text(text))
        return [s for s in sentences if s.strip()]
    except:
        sentences = re.split(r'[.!?]+', preprocess_text(text))
        return [s.strip() for s in sentences if s.strip()]

def clean_text(sentences):
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

def analyze_text(text_input):
    if not text_input or not text_input.strip():
        return pd.DataFrame({'Text': ['No text provided']})

    sentences = tokenize_sentences(text_input)
    if not sentences:
        return pd.DataFrame({'Text': ['No sentences found']})

    cleaned_sentences = clean_text(sentences)
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

def read_markdown_file(filename):
    try:
        path = resource_path(filename)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as file:
                return file.read()
    except:
        pass

    if 'welcome' in filename.lower():
        return "# Welcome to MarkLex\n\nMarkLex - Marketing Lexicon Creation"
    else:
        return "# Instructions\n\nUse the tabs above."

def markdown_to_html(text):
    html = text
    html = re.sub(r'^## (.*)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^# (.*)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'_(.*?)_', r'<em>\1</em>', html)
    html = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" target="_blank">\1</a>', html)
    html = html.replace('\n\n', '</p><p>')
    html = f'<p>{html}</p>'
    html = re.sub(r'<p>(<[^>]+>)</p>', r'\1', html)
    return html

# Load content
welcome_content = read_markdown_file("welcome.md")
instructions_content = read_markdown_file("instructions.md")

# Define UI
app_ui = ui.page_navbar(
    ui.nav_panel("MarkLex",
        ui.div(
            ui.HTML(markdown_to_html(welcome_content)),
            style="width: 60%; margin: auto;"
        )
    ),
    ui.nav_panel("Lexicon",
        ui.div(
            ui.div(
                ui.h3("Instructions"),
                ui.div(
                    ui.h4("Use a unigram or a bigram"),
                    ui.p("You can enter a unigram (i.e., a single term such as marketing, profit, privacy, etc.). You can also enter a bigram (i.e., two related terms separated by a space such as profit margin). The app will display terms most related to the search term along with the siilarity scores."),
                    ui.h4("Troubleshooting"),
                    ui.p("Consider different combinations if the app fails to return results for your search terms. Not all unigrams and bigrams may yield related terms, so we encourage experimenting with variations."),
                    style="background-color: #f8f9fa; border: 1px solid #e9ecef; padding: 1rem;"
                )
            ),
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_text("term",
                        ui.HTML("Add a term <span style='color: #B22222;'>(max 2 words)</span>:"),
                        placeholder="sustainability"
                    ),
                    ui.input_slider("n_words", "Number of words:",
                        value=20, min=10, max=100, step=5
                    ),
                    ui.input_action_button("run2", "Create Lexicon", class_="btn-primary"),
                    ui.download_button("downloadLex", "Download Lexicon")
                ),
                ui.output_data_frame("tab2")
            )
        )
    ),
    ui.nav_panel("Text Analysis",
        ui.div(
            ui.div(
                ui.h3("Instructions"),
                ui.p("This section of the app provides a demonstration of the use of the lexicon. Here, we have created a lexicon that comprises eight dimensions: Marketing, ESG, DEI, Risk & Security, Employee, AI & Data, Innovation, and Investor Focus. Paste your text in the space provided below and click Analyze Text. The output table will show you each sentence and the number of words associated with these dimensions. You can also output this table as a spreadsheet for further analysis."),
                style="background-color: #f8f9fa; padding: 1rem;"
            ),
            ui.div(
                ui.input_text_area("text_data1", "Input text:",
                    width="50%", height="150px"
                ),
                ui.div(
                    ui.input_action_button("run", "Analyze text", class_="btn-primary"),
                    ui.download_button("downloadData", "Download Data"),
                    style="display: inline-block; padding-left: 15px;"
                ),
                style="border: 1px solid #ddd; padding: 20px;"
            ),
            ui.output_data_frame("tab1")
        )
    ),
    title="MarkLex"
)

def server(input, output, session):
    text_output_data = reactive.Value(pd.DataFrame())
    lex_output_data = reactive.Value(pd.DataFrame())

    @reactive.Effect
    @reactive.event(input.run)
    async def analyze_handler():
        if input.text_data1():
            result = analyze_text(input.text_data1())
            text_output_data.set(result)

    @reactive.Effect
    @reactive.event(input.run2)
    async def lexicon_handler():
        if input.term():
            num_words = len(input.term().split())
            model_type = 'uni' if num_words == 1 else 'bi'
            result = create_lexicon_from_word(input.term(), input.n_words(), model_type)
            lex_output_data.set(result)

    @output
    @render.data_frame
    def tab1():
        df = text_output_data()
        if not df.empty:
            return render.DataGrid(df, width="100%", height="400px")
        return render.DataGrid(pd.DataFrame({'Message': ['Click Analyze text to start']}))

    @output
    @render.data_frame
    def tab2():
        df = lex_output_data()
        if not df.empty:
            return render.DataGrid(df, width="100%", height="400px")
        return render.DataGrid(pd.DataFrame({'Message': ['Enter a term and click Create Lexicon']}))

    @session.download(filename=lambda: f"text_analysis_{datetime.datetime.now().strftime('%Y%m%d')}.csv")
    def downloadData():
        return io.StringIO(text_output_data().to_csv(index=False))

    @session.download(filename=lambda: f"lexicon_{input.term()}_{datetime.datetime.now().strftime('%Y%m%d')}.csv")
    def downloadLex():
        return io.StringIO(lex_output_data().to_csv(index=False))

app = App(app_ui, server)

if __name__ == "__main__":
    print("=== Starting MarkLex App ===")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")

    # Determine if running in standalone mode
    is_standalone = hasattr(sys, '_MEIPASS') or 'MarkLex-Standalone' in sys.executable

    # Diagnostic information
    try:
        if hasattr(sys, '_MEIPASS'):
            print(f"PyInstaller temp directory: {sys._MEIPASS}")

            # List files in temp directory
            temp_files = os.listdir(sys._MEIPASS)
            print(f"Files in temp directory: {len(temp_files)} files")
            embedding_files = [f for f in temp_files if 'embedding' in f.lower()]
            if embedding_files:
                print(f"Embedding files found: {embedding_files}")
            else:
                print("❌ No embedding files found in temp directory!")
        else:
            print("Running in development mode (not PyInstaller)")
    except Exception as e:
        print(f"Could not check PyInstaller status: {e}")

    # List available files in current directory
    try:
        files = os.listdir('.')
        print(f"Files in current directory: {len(files)} files")

        # Check for critical files
        critical_files = ['Lexicon List.xlsx', 'welcome.md', 'instructions.md']
        for file in critical_files:
            if file in files:
                print(f"✓ Found: {file}")
            else:
                print(f"❌ Missing: {file}")

        # Check for embedding files
        embedding_files = [f for f in files if 'embedding' in f.lower() or f.endswith('.model')]
        if embedding_files:
            print(f"✓ Embedding files: {embedding_files}")
        else:
            print("❌ No embedding files found in current directory!")

    except Exception as e:
        print(f"Could not list files: {e}")

    # Choose host based on mode
    if is_standalone:
        host = "0.0.0.0"  # Accept connections from Electron wrapper
        print(f"Running in DESKTOP mode on http://{host}:8000")
    else:
        host = "127.0.0.1"  # Development mode
        print(f"Running in DEVELOPMENT mode on http://{host}:8000")

    print("=== App Starting ===")

    try:
        app.run(host=host, port=8000)
    except Exception as e:
        print(f"❌ Failed to start app: {e}")
        import traceback
        traceback.print_exc()
        raise
