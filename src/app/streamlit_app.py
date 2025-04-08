import os
import sys
import logging
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from io import BytesIO
import requests
from typing import List, Dict, Any, Optional
import base64

# Add parent directory to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.data_loader import DataLoader
from models.embeddings import EmbeddingGenerator
from models.vector_db import VectorDatabase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="AI Fashion Product Recommender",
    page_icon="👕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define base paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(ROOT_DIR, "ZARA_jackets_men.csv")
INDEX_DIR = os.path.join(ROOT_DIR, "indexes")

# Ensure index directory exists
os.makedirs(INDEX_DIR, exist_ok=True)

# Sidebar for app navigation
st.sidebar.title("AI Fashion Recommender")
st.sidebar.markdown("Search for similar fashion products using text, images, or both!")

search_mode = st.sidebar.selectbox(
    "Select Search Mode",
    ["Text Search", "Image Search", "Hybrid Search"],
)

# Function to load data and models
@st.cache_resource
def load_models():
    """Load data and initialize models."""
    # Load and preprocess data
    data_loader = DataLoader(DATA_PATH)
    data_loader.preprocess_data()
    
    # Initialize vector database
    vector_db = VectorDatabase()
    
    # Check if we have pre-computed indexes
    indexes_exist = os.path.exists(os.path.join(INDEX_DIR, "product_data.pkl"))
    
    if indexes_exist:
        # Load pre-computed indexes
        vector_db.load_indexes(INDEX_DIR)
        st.sidebar.success("✅ Loaded pre-computed embeddings and indexes")
    else:
        st.sidebar.warning("❌ No pre-computed embeddings found")
    
    return data_loader, vector_db

# Function to display product cards
def display_products(products: List[Dict[str, Any]]):
    """Display product cards in grid layout."""
    if not products:
        st.warning("No similar products found.")
        return
    
    # Display products in rows of 3
    cols = st.columns(3)
    
    for i, product in enumerate(products):
        col = cols[i % 3]
        with col:
            # Create a card-like container
            with st.container():
                st.subheader(product["product_name"])
                
                # Display image if available
                if product.get("primary_image_url"):
                    try:
                        response = requests.get(product["primary_image_url"], timeout=5)
                        if response.status_code == 200:
                            img = Image.open(BytesIO(response.content))
                            st.image(img, use_column_width=True)
                    except Exception as e:
                        st.error(f"Error loading image: {e}")
                
                # Product details
                st.write(product["details"])
                
                # Product link button
                if product.get("link"):
                    st.markdown(f"[View on ZARA]({product['link']})")
                
                # Show similarity score
                if "similarity_score" in product:
                    st.progress(min(float(product["similarity_score"]), 1.0))
                    st.caption(f"Similarity: {product['similarity_score']:.2f}")
                    
                st.markdown("---")

# Function to generate text embeddings
@st.cache_data
def get_text_embedding(text: str, embedding_generator: EmbeddingGenerator):
    """Generate embedding for text input."""
    embeddings = embedding_generator.generate_text_embeddings([text])
    return embeddings[0]

# Function to generate image embeddings
@st.cache_data
def get_image_embedding(image_file, embedding_generator: EmbeddingGenerator):
    """Generate embedding for image input."""
    # Convert uploaded file to image
    image = Image.open(image_file)
    
    # Convert image to base64
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    # Create instance for prediction
    instance = {
        "image": {
            "bytesBase64Encoded": img_str
        }
    }
    
    # Get embedding using the endpoint directly
    endpoint = embedding_generator.multimodal_model_name
    project_id = embedding_generator.project_id
    location = embedding_generator.location
    
    endpoint_path = f"projects/{project_id}/locations/{location}/publishers/google/models/{endpoint}"
    endpoint_obj = embedding_generator._create_endpoint(endpoint_path)
    
    response = endpoint_obj.predict(instances=[instance])
    embedding = np.array(response.predictions[0]["imageEmbedding"])
    
    return embedding

# Main function
def main():
    """Main application function."""
    # Load data and models
    data_loader, vector_db = load_models()
    
    # Check if Google Cloud credentials are set
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        st.sidebar.warning("⚠️ Google Cloud Project ID not set. Set GOOGLE_CLOUD_PROJECT environment variable.")
        project_id = st.sidebar.text_input("Enter Google Cloud Project ID:", "")
        if project_id:
            os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
    
    # Initialize embedding generator
    embedding_generator = None
    if project_id:
        try:
            embedding_generator = EmbeddingGenerator(project_id)
            st.sidebar.success(f"✅ Connected to Google Vertex AI - Project: {project_id}")
        except Exception as e:
            st.sidebar.error(f"❌ Error connecting to Google Vertex AI: {e}")
    
    # Main page header
    st.title("AI Fashion Product Recommender")
    st.markdown("""
    Find similar fashion products using AI-powered search.
    This demo uses Google Vertex AI for embeddings and FAISS for similarity search.
    """)
    
    # Check if we have pre-computed data
    if not vector_db.product_ids:
        st.warning("No product data loaded. Please generate embeddings first.")
        
        # Button to generate embeddings
        if st.button("Generate Embeddings") and embedding_generator:
            with st.spinner("Loading data and generating embeddings..."):
                # Get product data
                product_ids, texts, image_urls = data_loader.get_product_data()
                
                # Generate embeddings
                text_embeddings = embedding_generator.generate_text_embeddings(texts)
                st.success("✅ Text embeddings generated")
                
                image_embeddings = embedding_generator.generate_image_embeddings(image_urls)
                st.success("✅ Image embeddings generated")
                
                hybrid_embeddings = embedding_generator.generate_hybrid_embeddings(texts, image_urls)
                st.success("✅ Hybrid embeddings generated")
                
                # Build indexes
                vector_db.build_text_index(product_ids, text_embeddings)
                vector_db.build_image_index(product_ids, image_embeddings)
                vector_db.build_hybrid_index(product_ids, hybrid_embeddings)
                
                # Store product metadata
                product_metadata = data_loader.create_product_metadata()
                vector_db.store_product_data(product_metadata)
                
                # Save indexes
                vector_db.save_indexes(INDEX_DIR)
                
                st.success("✅ Embeddings generated and saved!")
                st.rerun()  # Rerun the app
        
        return
    
    # Display search interface based on selected mode
    if search_mode == "Text Search":
        st.header("Text-Based Search")
        
        # Text input
        query = st.text_input("Enter product description:", placeholder="e.g., leather jacket with zipper")
        
        # Number of results slider
        num_results = st.slider("Number of results:", 1, 10, 5)
        
        # Search button
        if st.button("Search") and query and embedding_generator:
            with st.spinner("Searching for similar products..."):
                # Generate embedding for query
                query_embedding = get_text_embedding(query, embedding_generator)
                
                # Get similar products
                similar_products = vector_db.search_by_text(query_embedding, k=num_results)
                
                # Display results
                st.subheader("Similar Products")
                display_products(similar_products)
    
    elif search_mode == "Image Search":
        st.header("Image-Based Search")
        
        # Image upload
        uploaded_file = st.file_uploader("Upload a jacket image:", type=["jpg", "jpeg", "png"])
        
        # Number of results slider
        num_results = st.slider("Number of results:", 1, 10, 5)
        
        # Search button
        if uploaded_file and embedding_generator:
            # Display the uploaded image
            st.image(uploaded_file, caption="Uploaded Image", width=300)
            
            if st.button("Search"):
                with st.spinner("Searching for similar products..."):
                    # Generate embedding for image
                    query_embedding = get_image_embedding(uploaded_file, embedding_generator)
                    
                    # Get similar products
                    similar_products = vector_db.search_by_image(query_embedding, k=num_results)
                    
                    # Display results
                    st.subheader("Similar Products")
                    display_products(similar_products)
    
    else:  # Hybrid Search
        st.header("Hybrid Search (Text + Image)")
        
        # Text input
        query = st.text_input("Enter product description:", placeholder="e.g., leather jacket with zipper")
        
        # Image upload
        uploaded_file = st.file_uploader("Upload a jacket image (optional):", type=["jpg", "jpeg", "png"])
        
        # Weighting slider
        text_weight = st.slider("Text vs. Image Weight:", 0.0, 1.0, 0.5, 
                               help="0 = Only Image, 1 = Only Text, 0.5 = Equal weight")
        
        # Number of results slider
        num_results = st.slider("Number of results:", 1, 10, 5)
        
        # Display uploaded image if any
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Image", width=300)
        
        # Search button
        if st.button("Search") and embedding_generator and (query or uploaded_file):
            with st.spinner("Searching for similar products..."):
                if query and uploaded_file:  # Both text and image
                    # Generate text embedding
                    text_embedding = get_text_embedding(query, embedding_generator)
                    
                    # Generate image embedding
                    image_embedding = get_image_embedding(uploaded_file, embedding_generator)
                    
                    # Normalize embeddings
                    text_embedding_norm = text_embedding / np.linalg.norm(text_embedding)
                    image_embedding_norm = image_embedding / np.linalg.norm(image_embedding)
                    
                    # Create hybrid embedding
                    hybrid_embedding = np.concatenate([
                        text_embedding_norm * text_weight,
                        image_embedding_norm * (1.0 - text_weight)
                    ])
                    # Normalize
                    hybrid_embedding = hybrid_embedding / np.linalg.norm(hybrid_embedding)
                    
                    # Get similar products
                    similar_products = vector_db.search_by_hybrid(hybrid_embedding, k=num_results)
                    
                elif query:  # Only text
                    query_embedding = get_text_embedding(query, embedding_generator)
                    similar_products = vector_db.search_by_text(query_embedding, k=num_results)
                    
                else:  # Only image
                    query_embedding = get_image_embedding(uploaded_file, embedding_generator)
                    similar_products = vector_db.search_by_image(query_embedding, k=num_results)
                
                # Display results
                st.subheader("Similar Products")
                display_products(similar_products)
                
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("AI Fashion Product Recommender")
    st.sidebar.caption("Using Google Vertex AI & FAISS")

if __name__ == "__main__":
    main()