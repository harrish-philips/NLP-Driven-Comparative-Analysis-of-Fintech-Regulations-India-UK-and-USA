# Plan: Streamlit FinRegNLP Analysis App

Build a multi-page Streamlit app that enables users to upload up to 4 regulatory documents, processes them through a complete NLP pipeline (text cleaning, frequency analysis, TF-IDF, BERT similarity, sentiment analysis), and displays interactive visualizations—all using pretrained models without external dependencies.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit Frontend                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ File Upload  │  │  Processing  │  │ Visualization │     │
│  │  (Sidebar)   │  │   Pipeline   │  │    Tabs       │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   Core Processing Modules                   │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │pdf_processor│  │frequency_    │  │similarity_   │      │
│  │    .py      │  │analyzer.py   │  │analyzer.py   │      │
│  └─────────────┘  └──────────────┘  └──────────────┘      │
│  ┌─────────────┐  ┌──────────────┐                         │
│  │sentiment_   │  │data_manager  │                         │
│  │analyzer.py  │  │    .py       │                         │
│  └─────────────┘  └──────────────┘                         │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   Pretrained Models Layer                   │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   spaCy     │  │    BERT      │  │    VADER     │      │
│  │en_core_web_ │  │bert-base-nli-│  │  Sentiment   │      │
│  │    sm       │  │ mean-tokens  │  │  Analyzer    │      │
│  └─────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Session State Cache                      │
│  • Uploaded documents  • Cleaned text  • Embeddings         │
│  • Similarity matrices • Sentiment scores • Visualizations  │
└─────────────────────────────────────────────────────────────┘
```

## System Design

### Component Architecture

**1. Frontend Layer (app.py)**
- Main Streamlit application
- File upload interface (sidebar, accepts up to 4 PDFs)
- Tab-based navigation (Frequency, Similarity, Sentiment)
- Progress indicators and status messages
- Visualization rendering

**2. Processing Modules**

**Module: pdf_processor.py**
- `extract_pdf_text(file_path)` - PyMuPDF text extraction
- `tokenize_text(text, nlp_model)` - spaCy tokenization
- `clean_text(text)` - Complete preprocessing pipeline
  - Lowercase conversion
  - Special character removal (regex)
  - POS-based lemmatization (TextBlob)
  - Stopword filtering (NLTK)

**Module: frequency_analyzer.py**
- `gen_freq(tokens)` - Word frequency calculation
- `create_word_df(word_freq)` - DataFrame generation
- `generate_wordcloud(data, **kwargs)` - WordCloud visualization
- `get_top_n_words(freq_series, n=20)` - Top N frequent words

**Module: similarity_analyzer.py**
- `compute_tfidf_similarity(docs, labels)` - TF-IDF vectorization + cosine similarity
- `compute_bert_similarity(docs, labels, model)` - BERT embeddings + cosine similarity
- `create_similarity_heatmap(matrix, labels, title)` - Heatmap visualization

**Module: sentiment_analyzer.py**
- `analyze_document_sentiment(text)` - VADER sentiment analysis
- `classify_sentences(sentences)` - Sentence-level classification
- `get_sentiment_distribution(classifications)` - Aggregate percentages
- `create_sentiment_chart(distributions)` - Pie chart visualization

**Module: data_manager.py**
- `initialize_session_state()` - Setup session state variables
- `cache_processed_data(key, value)` - Store results
- `get_cached_data(key)` - Retrieve cached results
- `clear_cache()` - Reset session state

### Data Flow

```
PDF Upload (Streamlit)
    │
    ▼
PyMuPDF Text Extraction
    │
    ▼
spaCy Tokenization (en_core_web_sm)
    │
    ▼
Text Cleaning Pipeline
    ├─ Lowercase
    ├─ Regex (remove special chars)
    ├─ POS Tagging (TextBlob)
    ├─ Lemmatization (WordNet)
    └─ Stopword Removal (NLTK)
    │
    ▼
Session State Cache (cleaned_texts)
    │
    ├──────────────┬──────────────┬──────────────┐
    │              │              │              │
    ▼              ▼              ▼              ▼
Frequency      TF-IDF        BERT          Sentiment
Analysis     Similarity   Similarity       Analysis
    │              │              │              │
    ▼              ▼              ▼              ▼
WordCloud      Heatmap       Heatmap      Pie Chart
Display        Display       Display       Display
```

## Step-by-Step Implementation Strategy

### Phase 1: Project Setup & Dependencies
1. Create project directory structure
2. Generate `requirements.txt` with all dependencies
3. Create placeholder module files
4. Setup spaCy model download instructions

### Phase 2: Core Processing Modules (Bottom-Up)

**Step 2.1: pdf_processor.py**
- Implement `extract_pdf_text()` using PyMuPDF
- Add error handling for corrupted PDFs
- Implement `clean_text()` with full preprocessing pipeline
- Add `@st.cache_data` for performance

**Step 2.2: frequency_analyzer.py**
- Port `gen_freq()` from notebook
- Create `generate_wordcloud()` with configurable parameters
- Add top-N word extraction
- Implement DataFrame formatting

**Step 2.3: similarity_analyzer.py**
- Implement TF-IDF vectorization with sklearn
- Add BERT model loading with caching
- Create embedding generation functions
- Build cosine similarity calculation
- Add heatmap generation with seaborn

**Step 2.4: sentiment_analyzer.py**
- Initialize VADER sentiment analyzer
- Implement sentence-level classification
- Calculate distribution percentages
- Create visualization functions

**Step 2.5: data_manager.py**
- Setup session state initialization
- Implement caching utilities
- Add data validation functions

### Phase 3: Streamlit Frontend

**Step 3.1: Main Application Structure (app.py)**
- Create basic Streamlit layout
- Add sidebar configuration
- Setup tab navigation structure

**Step 3.2: File Upload Interface**
- Implement file uploader (accept up to 4 PDFs)
- Add file validation (PDF format, size limits)
- Display uploaded file metadata
- Add "Process Documents" button

**Step 3.3: Processing Pipeline Integration**
- Connect upload to pdf_processor
- Add progress bars with `st.spinner()`
- Implement error handling and user feedback
- Cache processed results

**Step 3.4: Visualization Tabs**

**Tab 1: Frequency Analysis**
- Display top 20 words table
- Show WordCloud (400×330px)
- Add download buttons for data

**Tab 2: Similarity Analysis**
- Display TF-IDF similarity matrix (table + heatmap)
- Display BERT similarity matrix (table + heatmap)
- Add comparison insights

**Tab 3: Sentiment Analysis**
- Show sentiment distribution (pie charts)
- Display percentage breakdowns
- Add document-level comparisons
S
### Phase 4: Testing & Optimization

**Step 4.1: Model Loading Optimization**
- Implement `@st.cache_resource` for models
- Add lazy loading where appropriate
- Optimize memory usage

**Step 4.2: UI/UX Enhancements**
- Add custom styling (CSS)
- Implement responsive layouts
- Add tooltips and help text
- Improve error messages

**Step 4.3: Testing**
- Test with 1, 2, 3, and 4 documents
- Validate all analysis outputs
- Check edge cases (empty PDFs, corrupted files)
- Performance testing with large documents

## Module-by-Module Implementation Details

### Module 1: requirements.txt
```
streamlit>=1.30.0
PyMuPDF>=1.23.0
spacy>=3.7.0
textblob>=0.17.0
nltk>=3.8.0
sentence-transformers>=2.2.0
scikit-learn>=1.3.0
wordcloud>=1.9.0
matplotlib>=3.8.0
seaborn>=0.13.0
pandas>=2.1.0
numpy>=1.24.0
scipy>=1.11.0
```

### Module 2: pdf_processor.py

**Key Functions:**
```python
@st.cache_data(show_spinner=False)
def extract_pdf_text(file_bytes):
    """Extract text from PDF bytes using PyMuPDF"""
    
@st.cache_data(show_spinner=False)
def clean_text(text):
    """Complete text preprocessing pipeline"""
    # 1. Lowercase
    # 2. Remove special characters
    # 3. POS tagging with TextBlob
    # 4. Lemmatization
    # 5. Stopword removal
    
def tokenize_text(text, nlp_model):
    """Tokenize using spaCy"""
```

**Dependencies:** PyMuPDF, spaCy, TextBlob, NLTK, re

### Module 3: frequency_analyzer.py

**Key Functions:**
```python
def gen_freq(tokens):
    """Generate word frequency distribution"""
    
def create_word_df(word_freq, top_n=20):
    """Create DataFrame of top N words"""
    
@st.cache_data(show_spinner=False)
def generate_wordcloud(data, width=400, height=330, max_words=200):
    """Generate WordCloud visualization"""
    
def display_frequency_analysis(doc_name, cleaned_text, nlp_model):
    """Main function to display all frequency analysis"""
```

**Dependencies:** pandas, WordCloud, matplotlib

### Module 4: similarity_analyzer.py

**Key Functions:**
```python
@st.cache_data(show_spinner=False)
def compute_tfidf_similarity(docs_dict):
    """
    Args:
        docs_dict: {"Doc1": "cleaned text", "Doc2": "cleaned text", ...}
    Returns:
        similarity_df: DataFrame with similarity scores
    """
    
@st.cache_resource
def load_bert_model():
    """Load BERT model once and cache"""
    
@st.cache_data(show_spinner=False)
def compute_bert_similarity(_model, docs_dict):
    """
    Args:
        _model: SentenceTransformer model
        docs_dict: {"Doc1": "cleaned text", ...}
    Returns:
        similarity_df: DataFrame with similarity scores
    """
    
def create_similarity_heatmap(similarity_df, title):
    """Generate seaborn heatmap"""
```

**Dependencies:** scikit-learn, sentence-transformers, seaborn, matplotlib, scipy

### Module 5: sentiment_analyzer.py

**Key Functions:**
```python
@st.cache_resource
def load_vader_analyzer():
    """Initialize VADER sentiment analyzer"""
    
@st.cache_data(show_spinner=False)
def analyze_document_sentiment(text):
    """
    Returns:
        {
            "positive": percentage,
            "negative": percentage,
            "neutral": percentage
        }
    """
    
def create_sentiment_chart(sentiment_results):
    """Create pie chart visualization"""
    
def display_sentiment_comparison(docs_dict):
    """Display side-by-side sentiment comparison"""
```

**Dependencies:** NLTK (VADER), matplotlib

### Module 6: data_manager.py

**Key Functions:**
```python
def initialize_session_state():
    """Setup all session state variables"""
    if 'processed_docs' not in st.session_state:
        st.session_state.processed_docs = {}
    if 'cleaned_texts' not in st.session_state:
        st.session_state.cleaned_texts = {}
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False
        
def cache_processed_data(key, value):
    """Store data in session state"""
    
def get_cached_data(key, default=None):
    """Retrieve cached data"""
    
def clear_all_cache():
    """Reset all session state"""
```

**Dependencies:** streamlit

### Module 7: app.py (Main Application)

**Structure:**
```python
# Imports
# Page configuration
# Model loading functions (cached)
# Sidebar - File upload interface
# Main content area
#   - Tab 1: Frequency Analysis
#   - Tab 2: Similarity Analysis
#   - Tab 3: Sentiment Analysis
# Footer with instructions
```

**Key Features:**
- Multi-file uploader (max 4 files)
- Process button with progress tracking
- Tabbed interface for results
- Download buttons for results
- Clear cache button

## Clean Architecture Principles Applied

1. **Separation of Concerns**
   - Each module has single responsibility
   - UI logic separated from business logic
   - Data processing isolated from visualization

2. **Dependency Inversion**
   - Modules depend on abstractions (functions)
   - Models loaded via factory functions
   - Caching layer abstracts state management

3. **Modularity**
   - Each function performs one task
   - Functions are reusable and testable
   - Clear interfaces between modules

4. **Caching Strategy**
   - `@st.cache_resource` for models (singleton pattern)
   - `@st.cache_data` for data processing
   - Session state for user-specific data

5. **Error Handling**
   - Try-except blocks in all I/O operations
   - User-friendly error messages
   - Graceful degradation

## Constraints Compliance

✅ **Use pretrained models only**
- spaCy: `en_core_web_sm`
- BERT: `bert-base-nli-mean-tokens`
- VADER: Pretrained lexicon

✅ **No database**
- All data in session state
- No persistent storage

✅ **No cloud deployment**
- Local Streamlit app
- Local model storage

✅ **No authentication**
- Direct access to app
- No user management

## Further Considerations

### 1. Model Initialization Strategy
**Decision:** Preload all models at startup using `@st.cache_resource`
- **Pros:** Better UX, no delays during analysis
- **Cons:** Longer initial load time (~10-30 seconds)
- **Implementation:** Show progress spinner during model loading

### 2. Document Count Handling
**Decision:** Fixed limit of 4 documents with validation
- Disable upload after 4 files
- Show remaining slots in UI
- Clear validation messages

### 3. Layout Approach
**Decision:** Single-page app with tabbed interface
- **Reasoning:** Simpler navigation, all results visible
- **Alternative:** Multi-page would be better for 5+ analysis types

### 4. Data Persistence
**Decision:** Session state only, no temporary files
- **Reasoning:** Complies with "no database" constraint
- **Trade-off:** Data lost on page refresh (acceptable)

### 5. WordCloud Display
**Decision:** Fixed size (400×330px) as per notebook
- Maintains consistency with original analysis
- Compact display as required

### 6. Performance Optimization
- Cache model loading (saves 10-30s per reload)
- Cache text cleaning (saves 5-15s per document)
- Cache embeddings (saves 10-20s per analysis)
- Expected processing time: 30-60s for 4 documents (first run)

### 7. Error Scenarios
- Invalid PDF format → Show error, skip file
- Corrupted PDF → Show error, continue with others
- Empty text extraction → Warn user, exclude from analysis
- Memory limits → Process documents sequentially if needed

## Expected File Structure

```
finreg-nlp-app/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── modules/
│   ├── __init__.py
│   ├── pdf_processor.py        # Text extraction & cleaning
│   ├── frequency_analyzer.py   # Word frequency & WordCloud
│   ├── similarity_analyzer.py  # TF-IDF & BERT similarity
│   ├── sentiment_analyzer.py   # VADER sentiment analysis
│   └── data_manager.py         # Session state management
├── utils/
│   ├── __init__.py
│   └── config.py              # Constants & configuration
└── README.md                   # Setup instructions
```

## Next Steps

1. **Create project structure** - All directories and placeholder files
2. **Install dependencies** - Generate requirements.txt and install packages
3. **Download models** - spaCy en_core_web_sm, NLTK data
4. **Implement modules** - Bottom-up (processors → analyzers → app)
5. **Test incrementally** - Test each module before moving to next
6. **Integrate UI** - Build Streamlit interface last
7. **Final testing** - End-to-end with sample documents
