# MarkLex - Marketing Lexicon Creation

A Streamlit web application for creating marketing lexicons and analyzing text for various business dimensions using Word2Vec embeddings.

## Features

- **Lexicon Creation**: Generate word lists based on seed words using pre-trained Word2Vec embeddings
- **Text Analysis**: Analyze text documents for Marketing, ESG, Risk, and other business dimensions
- **Export Functionality**: Download results as CSV files
- **Free Hosting**: Runs on Streamlit Community Cloud at no cost

## Demo

🚀 **[Live App](https://your-app-name.streamlit.app)** *(URL will be available after deployment)*

## Installation

### Option 1: Run Locally

```bash
git clone https://github.com/your-username/marklex-streamlit.git
cd marklex-streamlit
pip install -r requirements_streamlit.txt
streamlit run streamlit_app.py
```

### Option 2: Deploy to Streamlit Community Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repository
5. Deploy!

## Usage

### Lexicon Creation
1. Go to the "Lexicon" tab
2. Enter a unigram (single word) or bigram (two words)
3. Choose the number of similar words to generate (10-100)
4. Click "Create Lexicon" to see semantically similar terms
5. Download results as CSV

### Text Analysis
1. Go to the "Text Analysis" tab
2. Paste your text in the input area
3. Click "Analyze Text" to see sentence-by-sentence analysis
4. Results show matches for different business dimensions
5. Download analysis as CSV

## Technical Details

- **Framework**: Streamlit
- **NLP**: Word2Vec embeddings (300MB models included)
- **Text Processing**: NLTK for tokenization and preprocessing
- **Data Handling**: Pandas for data manipulation
- **Deployment**: Streamlit Community Cloud (free tier)

## File Structure

```
├── streamlit_app.py          # Main Streamlit application
├── requirements_streamlit.txt # Python dependencies
├── embeddings_8              # Unigram Word2Vec model
├── embeddings_bi_grams       # Bigram Word2Vec model  
├── embeddings_8.wv.vectors.npy # Word vectors
├── embeddings_8.trainables.syn1neg.npy # Training data
├── Lexicon List.xlsx         # Default lexicon data
├── welcome.md               # Welcome page content
├── instructions.md          # Instructions content
└── README.md               # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

If you encounter any issues, please open an issue on GitHub or contact the maintainers.