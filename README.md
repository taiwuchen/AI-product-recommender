# AI Product Recommendation System

A multi-modal AI-based product recommendation system that uses vector search, embeddings, and RAG (Retrieval-Augmented Generation) to recommend fashion products based on text descriptions, images, or a combination of both.

## Features

- **Text-based Search**: Find products using natural language descriptions
- **Image-based Search**: Upload an image to find visually similar products
- **Hybrid Search**: Combine text and image queries for more refined results
- **RAG-powered Descriptions**: AI-generated recommendations and product descriptions
- **Vector Search**: Fast similarity search using FAISS
- **Multi-modal Embeddings**: Text and image embedding generation with TensorFlow and Google Vertex AI

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
   - Text embeddings are generated from product descriptions using TensorFlow's Universal Sentence Encoder or Google Vertex AI
   - Image embeddings are generated from product images using TensorFlow's MobileNetV2 or Google Vertex AI

3. **Vector Database**: Embeddings are stored in FAISS, a library for efficient similarity search

4. **Similarity Search**:
   - Text search matches products with similar descriptions
   - Image search matches products with similar visual characteristics
   - Hybrid search combines both approaches with adjustable weights

5. **RAG for Product Descriptions**: Generated descriptions analyze common features across recommended products

## Technologies Used

- **FAISS**: For vector similarity search
- **TensorFlow**: For embedding generation (fallback)
- **Google Vertex AI**: For embedding generation (when configured)
- **Streamlit**: For the web interface
- **Pillow**: For image processing
- **NumPy/Pandas**: For data manipulation

## Future Enhancements

- Implement a more sophisticated RAG system using LLMs (ChatGPT, PaLM, etc.)
- Add more product attributes for filtering
- Support for video-based search
- User preference tracking
- Integration with actual e-commerce platforms

## License

[MIT License](LICENSE)