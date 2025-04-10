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
from models.base_embedding import fix_certificate_verification
from models.text_embedding import TextEmbeddingGenerator
from models.image_embedding import ImageEmbeddingGenerator
from models.vector_db import VectorDatabase

# Fix SSL certificates at the start of the application
fix_certificate_verification()

# Constants
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

@st.cache_resource
def load_data():
    """Load and preprocess the product data."""
    loader = ProductDataLoader(DATA_PATH)
    return loader.preprocess_data(), loader

@st.cache_resource
def initialize_models():
    """Initialize the embedding generators and vector database."""
    # Create separate text and image embedding generators
    text_embedding_generator = TextEmbeddingGenerator(
        google_credentials_path=GOOGLE_CREDENTIALS_PATH
    )
    
    image_embedding_generator = ImageEmbeddingGenerator(
        google_credentials_path=GOOGLE_CREDENTIALS_PATH
    )
    
    vector_db = VectorDatabase()
    return text_embedding_generator, image_embedding_generator, vector_db

def build_or_load_indexes(df, text_embedding_generator, image_embedding_generator, vector_db):
    """Build or load vector indexes."""
    
    # Delete existing indexes to force rebuild
    import shutil
    if os.path.exists(INDEXES_DIR):
        print(f"Deleting existing indexes in {INDEXES_DIR}")
        try:
            # Delete all files in directory without removing directory
            for file_name in os.listdir(INDEXES_DIR):
                file_path = os.path.join(INDEXES_DIR, file_name)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            print("Existing indexes deleted successfully")
        except Exception as e:
            print(f"Error deleting indexes: {e}")
    
    # Create indexes directory if it doesn't exist
    os.makedirs(INDEXES_DIR, exist_ok=True)
    
    with st.spinner('Building indexes. This may take a while...'):
        try:
            print("Building new indexes")
            # Generate text embeddings
            print(f"Generating text embeddings for {len(df)} products")
            text_embeddings = text_embedding_generator.generate_batch_text_embeddings(df['text_for_embedding'].tolist())
            print(f"Generated text embeddings shape: {text_embeddings.shape}")
            
            # Generate image embeddings
            print(f"Generating image embeddings for {len(df)} products")
            image_embeddings = image_embedding_generator.generate_batch_image_embeddings(df['first_image_url'].tolist())
            print(f"Generated image embeddings shape: {image_embeddings.shape}")
            
            # Add embeddings to vector db
            print("Adding text embeddings to vector DB")
            vector_db.add_text_embeddings(text_embeddings, list(range(len(df))))
            print("Adding image embeddings to vector DB")
            vector_db.add_image_embeddings(image_embeddings, list(range(len(df))))
            
            # Store original text data for keyword filtering
            print("Setting product texts")
            vector_db.set_product_texts(df['text_for_embedding'].tolist())
            
            print(f"Final vector DB index stats: {vector_db.index_text.ntotal} text vectors, {len(vector_db.product_texts)} product texts")
            
            # Save indexes
            print(f"Saving indexes to {INDEXES_DIR}")
            vector_db.save_indices(INDEXES_DIR)
            st.success('Indexes built and saved successfully!')
        except Exception as e:
            st.error(f"Failed to build indexes: {e}")
            raise RuntimeError(f"Index building failed: {e}")

def generate_product_description(products: List[Dict], query: Optional[str] = None):
    """Generate a description for product recommendations."""
    if not products:
        return ""
        
    # Create description header
    if query:
        description = f"### Product Recommendations for '{query}'\n\n"
    else:
        description = "### Product Recommendations\n\n"
    
    # Extract common characteristics
    categories = []
    materials = []
    features = []
    
    for product in products:
        # Check that details is a string before calling .lower()
        if 'details' in product and product['details'] and isinstance(product['details'], str):
            details = product['details'].lower()
            product_name = product.get('name', '').lower()
            
            # Check if query terms exist in product details or name
            query_match = ""
            if query:
                query_terms = query.lower().split()
                matches = []
                for term in query_terms:
                    if term in details or term in product_name:
                        matches.append(term)
                
                if matches:
                    query_match = f"These products match your search for '{query}' because they contain {', '.join(matches)}. "
            
            # Extract possible categories
            if "bomber" in details or "bomber" in product_name:
                categories.append("bomber")
            elif "leather" in details or "leather" in product_name:
                categories.append("leather")
            elif "denim" in details or "denim" in product_name:
                categories.append("denim")
            elif "technical" in details or "technical" in product_name:
                categories.append("technical")
            
            # Extract possible materials
            if "cotton" in details:
                materials.append("cotton")
            elif "linen" in details:
                materials.append("linen")
            elif "suede" in details:
                materials.append("suede")
            elif "leather" in details:
                if "faux" in details:
                    materials.append("faux leather")
                else:
                    materials.append("leather")
            
            # Extract possible features
            if "zip" in details:
                features.append("zip closure")
            if "pocket" in details:
                features.append("pockets")
            if "hood" in details:
                features.append("hooded")
    
    # Get unique values
    categories = list(set(categories))
    materials = list(set(materials))
    features = list(set(features))
    
    # Build the description
    if query:
        description += query_match
        
    if categories:
        description += f"These recommendations focus on {', '.join(categories)} jackets "
        if materials:
            description += f"made with {', '.join(materials)} "
        description += "that might suit your style. "
    
    if features:
        description += f"Featured details include {', '.join(features)}. "
    
    description += "\n\nThese jackets share similar visual and functional characteristics, "
    description += "offering you a selection of complementary styles that align with your preferences."
    
    return description

def download_image(image_url):
    """Download image from URL."""
    try:
        # Add headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.zara.com/'
        }
        response = requests.get(image_url, stream=True, headers=headers)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        st.error(f"Error downloading image: {e}")
        return None

def display_product(product):
    """Display a product card."""
    col1, col2 = st.columns([1, 3])
    
    with col1:
        image = download_image(product['image_url'])
        if image:
            st.image(image, use_container_width=True)
    
    with col2:
        st.subheader(product['name'])
        st.write(product['details'])
        st.write(f"[View on ZARA]({product['link']})")

def main():
    """Main function to run the Streamlit app."""
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
        text_embedding_generator, image_embedding_generator, vector_db = initialize_models()
        model_name = os.environ.get("VERTEX_EMBEDDING_MODEL", "text-embedding-005")
        st.success(f"✅ Embedding models initialized successfully (using {model_name})")
    except Exception as e:
        st.error(f"❌ Failed to initialize embedding models: {e}")
        st.error("This application requires access to Google Vertex AI. Please check your credentials.")
        
        # Show credentials path for debugging
        st.info(f"GOOGLE_APPLICATION_CREDENTIALS: {GOOGLE_CREDENTIALS_PATH or 'Not set'}")
        
        instructions = """
        ### Google Cloud Authentication Error
        
        To fix this:
        1. Create a Google Cloud project and enable Vertex AI API
        2. Create a service account with Vertex AI User permissions
        3. Download the service account key as JSON
        4. Set GOOGLE_APPLICATION_CREDENTIALS environment variable to the path of the JSON file
        
        Example:
        ```
        export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your-key-file.json
        ```
        """
        st.markdown(instructions)
        st.stop()
    
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
        text_query = st.text_input("Enter your search query", "leather jacket with pockets", key="text_search_query")
        
        if st.button("Search by Text"):
            if text_query.strip():
                with st.spinner("Searching..."):
                    try:
                        # Generate text embedding for the query
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
        
        # Option 1: Upload an image
        uploaded_file = st.file_uploader("Choose an image of a jacket", type=["jpg", "jpeg", "png"], key="image_search_uploader")
        
        # Option 2: Enter image URL
        image_url = st.text_input("Or enter an image URL", key="image_search_url")
        
        if st.button("Search by Image"):
            image = None
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
            elif image_url.strip():
                image = download_image(image_url)
            
            if image:
                with st.spinner("Analyzing image and searching for similar products..."):
                    try:
                        # Preprocess image
                        img_array = image_embedding_generator.preprocess_image(image)
                        
                        # Generate image embedding
                        query_embedding = image_embedding_generator.image_model.predict(img_array)[0]
                        
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