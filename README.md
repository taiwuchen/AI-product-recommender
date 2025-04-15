# AI Product Recommender

An AI-powered product recommendation system for fashion items, featuring both text and image search. The system leverages vector search, retrieval-augmented generation (RAG), and large language models to generate creative, context-aware product descriptions and recommendations.

## Features

- **Text & Image Search:** Find similar products using either a text query or an image.
- **Retrieval-Augmented Generation (RAG):** Generates rich, creative product descriptions by combining LLMs with context from similar products.
- **Vector Database:** Fast similarity search using FAISS for both text and image embeddings.
- **Modern UI:** Streamlit-based web interface for interactive exploration.
- **Extensible:** Modular codebase for easy adaptation to other product domains.

## Project Structure

```
.
├── src/
│   ├── app/
│   │   └── streamlit_app.py      # Main Streamlit web app
│   ├── models/
│   │   ├── rag_generator.py      # RAG logic and LLM prompt construction
│   │   ├── vector_db.py          # Vector database using FAISS
│   │   ├── text_embedding.py     # Text embedding with Google Vertex AI
│   │   └── image_embedding.py    # Image embedding with CLIP
│   └── utils/
│       └── data_loader.py        # Product data loading and preprocessing
├── ZARA_jackets_men.csv          # Example product dataset
├── requirements.txt              # Python dependencies
├── vector_search_rag_demo.ipynb  # Notebook demo of vector search & RAG
```

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd AI-product-recommender
```

### 2. Install dependencies

It's recommended to use a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure API Keys

- **Google Vertex AI:** Required for text embeddings. Set up a Google Cloud project and authenticate using the Google Cloud CLI:
  ```
  gcloud auth application-default login
  ```
This will open a browser window for you to log in with your Google account. The credentials will be stored locally and used automatically by the application.

- **OpenRouter API Key:** For LLM product description generation. Set your API key in `src/app/streamlit_app.py` or via environment variable.

### 4. Prepare Data

Ensure `ZARA_jackets_men.csv` is present in the project root. You can adapt the loader for your own product data.

## Running the App

```bash
streamlit run src/app/streamlit_app.py
```

The app will:
- Load and preprocess product data
- Build or load vector indexes for text and image search
- Provide a web UI for searching and viewing recommendations

## Example Usage

- **Text Search:** Enter a product description or keywords to find similar items.
- **Image Search:** Upload an image to find visually similar products.
- The app generates a creative product description and key features using RAG and LLMs.

## Notebooks

- `vector_search_rag_demo.ipynb`: Demonstrates vector search and RAG concepts in a simplified setting.

## Customization

- Swap in your own product data by updating the CSV and loader.
- Adjust the LLM prompt or RAG logic in `rag_generator.py` for different product domains.

## Dependencies

- Streamlit
- FAISS
- Google Cloud Vertex AI
- Transformers (for CLIP)
- Pillow, Pandas, Numpy, Scikit-learn, etc.

See `requirements.txt` for the full list.

## License

MIT License