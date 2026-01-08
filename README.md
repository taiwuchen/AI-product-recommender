# Product Recommender

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
├── ZARA_jackets_men.csv          # Sample product dataset
├── requirements.txt              # Python dependencies
├── .env.example                  # Example environment variables
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/taiwuchen/semantic-product-search.git
cd semantic-product-search
```

### 2. Install dependencies

It's recommended to use a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy the example environment file and fill in your API keys:

```bash
cp .env.example .env
```

Required environment variables:
- `OPENROUTER_API_KEY`: Your OpenRouter API key for LLM product description generation
- `GOOGLE_APPLICATION_CREDENTIALS`: (Optional) Path to Google Cloud service account JSON file

For Google Vertex AI text embeddings, you can either:
1. Set `GOOGLE_APPLICATION_CREDENTIALS` to your service account JSON path, or
2. Use default credentials via Google Cloud CLI:
   ```bash
   gcloud auth application-default login
   ```

### 4. Prepare Data

A sample dataset (`ZARA_jackets_men.csv`) is included. To use your own data, provide a CSV with columns: `product_name`, `link`, `product_images`, `details` and update `DATA_PATH` in `streamlit_app.py`.

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