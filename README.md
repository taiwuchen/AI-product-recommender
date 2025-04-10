# AI Product Recommendation System

A product recommendation system that uses vector search and embeddings to recommend fashion products based on text descriptions or images.

## Features

- **Text-based Search**: Find products using natural language descriptions
- **Image-based Search**: Upload an image to find visually similar products
- **Vector Search**: Fast similarity search using FAISS
- **Multi-modal Embeddings**: Text and image embedding generation with Google Vertex AI

## Project Structure

```
AI-product-recommender/
├── ZARA_jackets_men.csv      # Sample product dataset
├── requirements.txt          # Python dependencies
├── indexes/                  # Directory for FAISS indexes
└── src/
    ├── app/
    │   └── streamlit_app.py  # Streamlit web application
    ├── models/
    │   ├── embeddings.py     # Text and image embedding generation
    │   └── vector_db.py      # Vector database using FAISS
    └── utils/
        └── data_loader.py    # Data loading and preprocessing
```

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AI-product-recommender.git
cd AI-product-recommender
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Set up Google Cloud credentials for Vertex AI:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credentials.json"
export GOOGLE_CLOUD_PROJECT="authentic-arch-456221-j3"
export VERTEX_EMBEDDING_MODEL="text-embedding-large-exp-03-07"
```

## Running the Application

Start the Streamlit web application:
```bash
cd src/app
streamlit run streamlit_app.py
```

## How It Works

1. **Data Loading**: The system loads product data from a CSV file containing product titles, descriptions, and image URLs.

2. **Embedding Generation**:
   - Text embeddings are generated from product descriptions using Google Vertex AI
   - Image embeddings are generated from product images using TensorFlow's MobileNetV2 or Google Vertex AI

3. **Vector Database**: Embeddings are stored in FAISS, a library for efficient similarity search

4. **Similarity Search**:
   - Text search matches products with similar descriptions
   - Image search matches products with similar visual characteristics

## Technologies Used

- **FAISS**: For vector similarity search
- **TensorFlow**: For image embedding generation (fallback)
- **Google Vertex AI**: For text embedding generation
- **Streamlit**: For the web interface
- **Pillow**: For image processing
- **NumPy/Pandas**: For data manipulation

## License

[MIT License](LICENSE)