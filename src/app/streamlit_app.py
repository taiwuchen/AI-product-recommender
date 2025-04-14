import os
import sys
import streamlit as st
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from typing import List, Dict, Optional

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import ProductDataLoader
from models.text_embedding import TextEmbeddingGenerator
from models.image_embedding import ImageEmbeddingGenerator
from models.vector_db import VectorDatabase

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                         'ZARA_jackets_men.csv')
INDEXES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'indexes')
GOOGLE_CREDENTIALS_PATH = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')

# Set up page configuration
st.set_page_config(
    page_title="AI Product Recommender",
    page_icon="👕",
    layout="wide"
)

# Global variables for models
text_embedding_generator = None
image_embedding_generator = None
vector_db = None

@st.cache_resource
def load_data():
    loader = ProductDataLoader(DATA_PATH)
    return loader.preprocess_data(), loader

@st.cache_resource
def initialize_models():
    global text_embedding_generator, image_embedding_generator, vector_db
    
    # Create separate text and image embedding generators
    text_embedding_generator = TextEmbeddingGenerator(
        google_credentials_path=GOOGLE_CREDENTIALS_PATH
    )

    image_embedding_generator = ImageEmbeddingGenerator()
    
    vector_db = VectorDatabase()
    return text_embedding_generator, image_embedding_generator, vector_db

def build_or_load_indexes(df, text_embedding_generator, image_embedding_generator, vector_db):
    # Delete existing indexes to force rebuild
    import shutil
    if os.path.exists(INDEXES_DIR):
        try:
            # Delete all files in directory without removing directory
            for file_name in os.listdir(INDEXES_DIR):
                file_path = os.path.join(INDEXES_DIR, file_name)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting indexes: {e}")
    
    # Create indexes directory if it doesn't exist
    os.makedirs(INDEXES_DIR, exist_ok=True)
    
    with st.spinner('Building indexes. This may take a while...'):
        try:
            # Generate text embeddings - using the updated function with list input
            text_embeddings = text_embedding_generator.generate_text_embedding(df['text_for_embedding'].tolist())
            print(f"Generated text embeddings shape: {text_embeddings.shape}")
            
            # Generate image embeddings
            image_embeddings = image_embedding_generator.generate_batch_image_embeddings(df['first_image_url'].tolist())
            print(f"Generated image embeddings shape: {image_embeddings.shape}")
            
            # Add embeddings to vector db
            vector_db.add_text_embeddings(text_embeddings, list(range(len(df))))
            vector_db.add_image_embeddings(image_embeddings, list(range(len(df))))
            
            # Store original text data for keyword filtering
            vector_db.set_product_texts(df['text_for_embedding'].tolist())
            
            # Save indexes
            vector_db.save_indices(INDEXES_DIR)
            st.success('Indexes built and saved successfully!')
        except Exception as e:
            st.error(f"Failed to build indexes: {e}")
            raise RuntimeError(f"Index building failed: {e}")

def generate_product_description(products: List[Dict], query: Optional[str] = None):
    if not products:
        return ""
    
    details_list = []
    categories = []
    materials = []
    features = []
    
    for product in products:
        name = product.get("name", "")
        details = product.get("details", "")
        details_list.append(f"Product Name: {name}\nProduct Details: {details}")
        if details and isinstance(details, str):
            details_lower = details.lower()
            product_name_lower = name.lower()
            if "bomber" in details_lower or "bomber" in product_name_lower:
                categories.append("bomber")
            elif "leather" in details_lower or "leather" in product_name_lower:
                categories.append("leather")
            elif "denim" in details_lower or "denim" in product_name_lower:
                categories.append("denim")
            elif "technical" in details_lower or "technical" in product_name_lower:
                categories.append("technical")
            if "cotton" in details_lower:
                materials.append("cotton")
            elif "linen" in details_lower:
                materials.append("linen")
            elif "suede" in details_lower:
                materials.append("suede")
            elif "leather" in details_lower:
                if "faux" in details_lower:
                    materials.append("faux leather")
                else:
                    materials.append("leather")
            if "zip" in details_lower:
                features.append("zip closure")
            if "pocket" in details_lower:
                features.append("pockets")
            if "hood" in details_lower:
                features.append("hooded")
    
    categories = list(set(categories))
    materials = list(set(materials))
    features = list(set(features))
    
    aggregated_info = "\n".join(details_list)
    extra_info = ""
    if categories:
        extra_info += "Categories: " + ", ".join(categories) + "\n"
    if materials:
        extra_info += "Materials: " + ", ".join(materials) + "\n"
    if features:
        extra_info += "Features: " + ", ".join(features) + "\n"
    
    prompt = ""
    if query:
        prompt += f"User Query: {query}\n\n"
    prompt += "Product Information:\n" + aggregated_info + "\n\n"
    if extra_info:
        prompt += "Additional Details:\n" + extra_info + "\n\n"
    prompt += "Based on the above information, generate a Top 5 comprehensive, creative, and engaging product description and recommendation."
    
    import os
    import requests
    import json
    API_KEY = "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"
    
    headers = {
        "Authorization": "Bearer " + API_KEY,
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "deepseek/deepseek-chat-v3-0324:free",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, data=json.dumps(payload))
    except Exception as e:
        return f"Error during API call: {str(e)}"
    
    if response.status_code == 200:
        try:
            response_json = response.json()
            result_text = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
            return result_text if result_text else "No description generated."
        except Exception as e:
            return "Error parsing API response."
    else:
        return f"Error: {response.status_code} - {response.text}"

def display_product(product):
    col1, col2 = st.columns([1, 3])
    
    with col1:
        try:
            image = image_embedding_generator.download_image(product['image_url'], convert_to_rgb=True, referer='https://www.zara.com/')
            if image:
                st.image(image, use_container_width=True)
            else:
                st.error("Failed to load image")
        except Exception as e:
            st.error(f"Error downloading image: {e}")
    
    with col2:
        st.subheader(product['name'])
        st.write(product['details'])
        st.write(f"[View on ZARA]({product['link']})")

def main():
    st.title("AI Product Recommendation System")
    st.write("Search for fashion products using text or image!")
    
    # Ensure indexes directory exists
    os.makedirs(INDEXES_DIR, exist_ok=True)
    
    # Load data
    try:
        df, loader = load_data()
        st.success("✅ Product data loaded successfully")
    except Exception as e:
        st.error(f"❌ Failed to load product data: {e}")
        st.stop()
    
    # Initialize models
    try:
        # Use global variables
        global text_embedding_generator, image_embedding_generator, vector_db
        text_embedding_generator, image_embedding_generator, vector_db = initialize_models()
        model_name = os.environ.get("VERTEX_EMBEDDING_MODEL", "text-embedding-005")
        st.success(f"✅ Embedding models initialized successfully (using {model_name})")
    except Exception as e:
        st.error(f"❌ Failed to initialize embedding models: {e}")
        st.error("This application requires access to Google Vertex AI. Please check your credentials.")
    
    # Build or load indexes
    try:
        build_or_load_indexes(df, text_embedding_generator, image_embedding_generator, vector_db)
    except Exception as e:
        st.error(f"❌ Failed to build indexes: {e}")
        st.error("Cannot continue without properly built indexes.")
        st.stop()
    
    # Create tabs for different search modes
    tab1, tab2 = st.tabs(["Text Search", "Image Search"])
    
    with tab1:
        st.header("Search by Text")
        text_query = st.text_input("Enter your search query", "", key="text_search_query")
        
        if st.button("Search by Text"):
            if text_query.strip():
                with st.spinner("Searching..."):
                    try:
                        query_embedding = text_embedding_generator.generate_text_embedding(text_query)
                        
                        # Search by text with keyword boosting enabled
                        distances, indices = vector_db.search_by_text(
                            query_embedding, k=5, query_text=text_query, keyword_boost=True)
                        
                        # Display results
                        st.subheader("Results")
                        
                        # Get product details
                        products = loader.get_product_details(indices[0].tolist())
                        
                        if not products:
                            st.warning("No products found matching your query.")
                        else:
                            # Display product description
                            st.markdown(generate_product_description(products, text_query))
                            
                            # Display product cards
                            for product in products:
                                st.divider()
                                display_product(product)
                    except Exception as e:
                        st.error(f"❌ Search failed: {e}")
    
    with tab2:
        st.header("Search by Image")
        
        st.info("Please provide a single image using ONE of the methods below:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Option 1: Upload an image
            uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="image_search_uploader")
        
        with col2:
            # Option 2: Enter image URL
            image_url = st.text_input("Or enter an image URL", key="image_search_url")
        
        if st.button("Search by Image"):
            image = None
            
            # Check if both methods are used and warn the user
            if uploaded_file is not None and image_url.strip():
                st.warning("You've provided both an uploaded file and a URL. Only the uploaded file will be used.")
                
            # Process the uploaded file first if available
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
            # Otherwise try the URL
            elif image_url.strip():
                try:
                    image = image_embedding_generator.download_image(image_url, convert_to_rgb=True, referer='https://www.zara.com/')
                    if image is None:
                        st.error("Failed to load image from URL")
                except Exception as e:
                    st.error(f"Error downloading image: {e}")
            else:
                st.warning("Please provide an image by uploading a file or entering a URL.")
            
            if image:
                with st.spinner("Analyzing image and searching for similar products..."):
                    try:
                        # Generate image embedding using CLIP
                        query_embedding = image_embedding_generator.generate_embedding_from_pil_image(image)
                        
                        # Search by image
                        distances, indices = vector_db.search_by_image(query_embedding, k=5)
                        
                        # Display results
                        st.subheader("Results")
                        
                        # Get product details
                        products = loader.get_product_details(indices[0].tolist())
                        
                        if not products:
                            st.warning("No products found matching your image.")
                        else:
                            # Display product description
                            st.markdown(generate_product_description(products, "your image"))
                            
                            # Display product cards
                            for product in products:
                                st.divider()
                                display_product(product)
                    except Exception as e:
                        st.error(f"❌ Image search failed: {e}")

if __name__ == "__main__":
    main()
