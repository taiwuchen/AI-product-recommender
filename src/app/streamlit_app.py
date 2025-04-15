import os
import sys
import streamlit as st
import requests
import json
from PIL import Image
from typing import Dict, Optional, List

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import ProductDataLoader
from models.text_embedding import TextEmbeddingGenerator
from models.image_embedding import ImageEmbeddingGenerator
from models.vector_db import VectorDatabase
from models.rag_generator import RAGGenerator

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                         'ZARA_jackets_men.csv')
INDEXES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'indexes')
GOOGLE_CREDENTIALS_PATH = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
API_KEY = os.environ.get('OPENROUTER_API_KEY')

# Set up page configuration
st.set_page_config(
    page_title="AI Product Recommender",
    page_icon="👕",
    layout="wide"
)

# --- Inject custom CSS from style.css ---
css_path = os.path.join(os.path.dirname(__file__), "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## 👕 AI Product Recommender")
    st.markdown(
        "Welcome! This app helps you find fashion products using **text** or **image** search powered by AI.\n\n"
        "- Use the **Text Search** tab to describe what you want.\n"
        "- Use the **Image Search** tab to upload or link to a product image.\n\n"
        "Results are enhanced with AI-generated descriptions."
    )
    st.markdown("---")
    st.markdown("Made with ❤️ using [Streamlit](https://streamlit.io/)")
    st.markdown("[GitHub Repo](https://github.com/taiwuchen/AI-product-recommender)")

# Global variables for models
text_embedding_generator = None
image_embedding_generator = None
vector_db = None
rag_generator = None

@st.cache_resource
def load_data():
    loader = ProductDataLoader(DATA_PATH)
    return loader.preprocess_data(), loader

@st.cache_resource
def initialize_models():
    global text_embedding_generator, image_embedding_generator, vector_db, rag_generator
    
    # Create separate text and image embedding generators
    text_embedding_generator = TextEmbeddingGenerator(
        google_credentials_path=GOOGLE_CREDENTIALS_PATH
    )

    image_embedding_generator = ImageEmbeddingGenerator()
    
    vector_db = VectorDatabase()
    
    # Load data for the RAG generator
    _, loader = load_data()
    
    # Initialize the RAG generator with vector_db, loader, and API key
    rag_generator = RAGGenerator(vector_db=vector_db, product_loader=loader, api_key=API_KEY)
    
    return text_embedding_generator, image_embedding_generator, vector_db, rag_generator

def build_or_load_indexes(df, text_embedding_generator, image_embedding_generator, vector_db):
    # Delete existing indexes to force rebuild
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
            image_embeddings = image_embedding_generator.generate_batch_image_embeddings(df['image_url'].tolist())
            print(f"Generated image embeddings shape: {image_embeddings.shape}")
            
            # Add embeddings to vector db
            vector_db.add_text_embeddings(text_embeddings, list(range(len(df))))
            vector_db.add_image_embeddings(image_embeddings, list(range(len(df))))
            
            # Store original text data for keyword filtering
            vector_db.set_product_texts(df['text_for_embedding'].tolist())
            
            # Save indexes
            vector_db.save_indices(INDEXES_DIR)
            st.success('✅ Indexes built and saved successfully!')
        except Exception as e:
            st.error(f"Failed to build indexes: {e}")
            raise RuntimeError(f"Index building failed: {e}")

def display_product(product, similar_products=None, query=None):
    # Product Card Style
    card = st.container()
    with card:
        col1, col2 = st.columns([1, 2.5])
        with col1:
            try:
                image = image_embedding_generator.download_image(product['image_url'], convert_to_rgb=True, referer='https://www.zara.com/')
                if image:
                    st.image(image, use_container_width="always", caption=product['name'])
                else:
                    st.error("Failed to load image")
            except Exception as e:
                st.error(f"Error downloading image: {e}")

        with col2:
            st.markdown(f"### {product['name']}")
            st.markdown(f"[View on ZARA]({product['link']})", help="Open product page in a new tab")
            with st.expander("Show product details"):
                st.write(product['details'])

            # Always use RAG generator for product descriptions
            with st.spinner("Generating enhanced description..."):
                if similar_products is None:
                    similar_products = []
                description = rag_generator.generate_product_description_with_rag(
                    product=product,
                    similar_products=similar_products,
                    query=query
                )
                st.markdown("**AI-Enhanced Description:**")
                st.markdown(description)
    st.markdown("---")

def main():
    st.markdown(
        "<h1 class='app-title'>👕 AI Product Recommendation System</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div class='app-desc'>"
        "Search for fashion products using <b>text</b> or <b>image</b>!<br>"
        "Powered by AI and RAG."
        "</div>",
        unsafe_allow_html=True
    )
    st.markdown("")

    # Load data
    try:
        df, loader = load_data()
        st.success("✅ Product data loaded successfully")
    except Exception as e:
        st.error(f"❌ Failed to load product data: {e}")
        st.stop()

    # Initialize models
    try:
        global text_embedding_generator, image_embedding_generator, vector_db, rag_generator
        text_embedding_generator, image_embedding_generator, vector_db, rag_generator = initialize_models()
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

    # --- MAIN TABS ---
    tab1, tab2 = st.tabs(
        [
            "🔤 Text Search",
            "🖼️ Image Search"
        ]
    )

    with tab1:
        st.header("🔤 Search by Text")
        st.markdown("Describe the product you want. For example: *'black bomber jacket with zipper pockets'*")
        text_query = st.text_input(
            "Enter your search query",
            "",
            key="text_search_query",
            help="Describe the product you are looking for (e.g., color, style, features)."
        )

        if st.button("Search by Text", help="Click to search for products matching your description."):
            if text_query.strip():
                with st.spinner("Searching and generating description..."):
                    try:
                        query_embedding = text_embedding_generator.generate_text_embedding(text_query)
                        distances, indices = vector_db.search_by_text(
                            query_embedding, k=5, query_text=text_query, keyword_boost=True)
                        st.subheader("Results")
                        products = loader.get_product_details(indices[0].tolist())
                        if not products:
                            st.warning("No products found matching your query.")
                        else:
                            for i, product in enumerate(products):
                                display_product(product, similar_products=products, query=text_query)
                    except Exception as e:
                        st.error(f"❌ Search failed: {e}")

    with tab2:
        st.header("🖼️ Search by Image")
        st.markdown("Find similar products by uploading an image or providing an image URL.")

        st.info("Please provide a single image using **one** of the methods below:")

        col1, col2 = st.columns(2)

        with col1:
            uploaded_file = st.file_uploader(
                "Upload an image",
                type=["jpg", "jpeg", "png"],
                key="image_search_uploader",
                help="Upload a product image from your device."
            )

        with col2:
            image_url = st.text_input(
                "Or enter an image URL",
                key="image_search_url",
                help="Paste a direct link to a product image."
            )

        if st.button("Search by Image", help="Click to search for products similar to your image."):
            image = None

            if uploaded_file is not None and image_url.strip():
                st.warning("You've provided both an uploaded file and a URL. Only the uploaded file will be used.")

            if uploaded_file is not None:
                image = Image.open(uploaded_file)
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
                        query_embedding = image_embedding_generator.generate_embedding_from_pil_image(image)
                        distances, indices = vector_db.search_by_image(query_embedding, k=5)
                        st.subheader("Results")
                        products = loader.get_product_details(indices[0].tolist())
                        if not products:
                            st.warning("No products found matching your image.")
                        else:
                            for product in products:
                                display_product(product)
                    except Exception as e:
                        st.error(f"❌ Image search failed: {e}")

    # --- FOOTER ---
    st.markdown(
        "<hr class='app-footer-hr'>"
        "<div class='app-footer'>"
        "AI Product Recommender &copy; 2025 &mdash; Built with Streamlit | "
        "<a href='https://github.com/your-repo-link' target='_blank'>GitHub</a>"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
