import os
import sys
import streamlit as st
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from typing import List, Dict, Optional
import json  # Added for JSON handling

# Set OpenRouter API key
os.environ['OPENROUTER_API_KEY'] = os.environ.get('OPENROUTER_API_KEY')

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
    # Check if indexes already exist
    if os.path.exists(os.path.join(INDEXES_DIR, 'text_index.faiss')) and \
       os.path.exists(os.path.join(INDEXES_DIR, 'image_index.faiss')):
        with st.spinner('Loading existing indexes...'):
            vector_db.load_indices(INDEXES_DIR)
            st.success('Indexes loaded successfully!')
    else:
        with st.spinner('Building indexes. This may take a while...'):
            # Generate text embeddings
            text_embeddings = text_embedding_generator.generate_batch_text_embeddings(df['text_for_embedding'].tolist())
            
            # Generate image embeddings
            image_embeddings = image_embedding_generator.generate_batch_image_embeddings(df['first_image_url'].tolist())
            
            # Add embeddings to vector db
            vector_db.add_text_embeddings(text_embeddings, list(range(len(df))))
            vector_db.add_image_embeddings(image_embeddings, list(range(len(df))))
            
            # Save indexes
            os.makedirs(INDEXES_DIR, exist_ok=True)
            vector_db.save_indices(INDEXES_DIR)
            st.success('Indexes built and saved successfully!')

def generate_rag_description(products: List[Dict], query: Optional[str] = None):
    """Generate an AI-powered description using RAG approach."""
    # Enhanced RAG implementation using LLM
    
    if not products:
        return ""
    
    # Extract information from products for context
    product_info = []
    for product in products:
        if 'name' in product and 'details' in product and product['details'] and isinstance(product['details'], str):
            product_info.append({
                "name": product['name'],
                "details": product['details']
            })
    
    # Check if OpenRouter API key is available for Gemini
    openrouter_api_key = os.environ.get('OPENROUTER_API_KEY')
    
    if product_info and openrouter_api_key:
        try:
            import requests
            
            # Create context from product information
            context = "Based on the following products:\n"
            for i, info in enumerate(product_info):
                context += f"{i+1}. {info['name']}: {info['details']}\n"
            
            # Create prompt for the LLM
            if query:
                prompt = f"{context}\n\nGenerate a concise recommendation paragraph for the search query '{query}'. Analyze common characteristics like product category, materials, and features. Mention how these products might suit the user's style and preferences."
            else:
                prompt = f"{context}\n\nGenerate a concise recommendation paragraph for these products. Analyze common characteristics like product category, materials, and features. Mention how these products might suit the user's style and preferences."
            
            # Call OpenRouter API with Gemini
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openrouter_api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://ai-product-recommender.app",  # Replace with your actual site URL
                    "X-Title": "AI Product Recommender",  # Replace with your actual site name
                },
                json={
                    "model": "google/gemini-2.5-pro-exp-03-25:free",  # Using Gemini model
                    "messages": [
                        {"role": "system", "content": "You are a fashion retail assistant providing helpful product recommendations."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 300,
                    "temperature": 0.7
                }
            )
            
            response_data = response.json()
            
            # Extract and format the response
            if response.status_code == 200 and "choices" in response_data and response_data["choices"]:
                title = f"### AI-Generated Recommendation for '{query}'" if query else "### AI-Generated Recommendation"
                description = response_data["choices"][0]["message"]["content"]
                return f"{title}\n\n{description}"
                
        except Exception as e:
            st.warning(f"Gemini API error: {e}. Trying alternative Gemini approach as fallback.")
    
    # If the first attempt with Gemini failed, try an alternative approach with Gemini
    openrouter_api_key = os.environ.get('OPENROUTER_API_KEY', os.environ.get('OPENROUTER_API_KEY'))
    if product_info and openrouter_api_key:
        try:
            import requests
            import json
            
            # Create context from product information
            context = "Based on the following products:\n"
            for i, info in enumerate(product_info):
                context += f"{i+1}. {info['name']}: {info['details']}\n"
            
            # Create prompt for the LLM
            if query:
                prompt = f"{context}\n\nGenerate a concise recommendation paragraph for the search query '{query}'. Analyze common characteristics like product category, materials, and features. Mention how these products might suit the user's style and preferences."
            else:
                prompt = f"{context}\n\nGenerate a concise recommendation paragraph for these products. Analyze common characteristics like product category, materials, and features. Mention how these products might suit the user's style and preferences."
            
            # Call OpenRouter API with Gemini as a fallback
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openrouter_api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://ai-product-recommender.app",
                    "X-Title": "AI Product Recommender",
                },
                data=json.dumps({
                    "model": "google/gemini-2.5-pro-exp-03-25:free",
                    "messages": [
                        {"role": "system", "content": "You are a fashion retail assistant providing helpful product recommendations."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 300,
                    "temperature": 0.7
                })
            )
            
            response_data = response.json()
            
            # Extract and format the response
            if response.status_code == 200 and "choices" in response_data and response_data["choices"]:
                title = f"### AI-Generated Recommendation for '{query}'" if query else "### AI-Generated Recommendation"
                description = response_data["choices"][0]["message"]["content"]
                return f"{title}\n\n{description}"
                
        except Exception as e:
            st.warning(f"Alternative LLM API error: {e}. Using template-based description instead.")
    
    
            
    # Fallback to template-based approach if LLM fails or API key not available
    if query:
        description = f"### AI-Generated Recommendation for '{query}'\n\n"
    else:
        description = "### AI-Generated Recommendation\n\n"
    
    # Extract common characteristics
    categories = []
    materials = []
    features = []
    
    for product in products:
        # Check that details is a string before calling .lower()
        if 'details' in product and product['details'] and isinstance(product['details'], str):
            details = product['details'].lower()
            
            # Extract possible categories
            if "bomber" in details:
                categories.append("bomber")
            elif "leather" in details:
                categories.append("leather")
            elif "denim" in details:
                categories.append("denim")
            elif "technical" in details:
                categories.append("technical")
            
            # Extract possible materials
            if "cotton" in details:
                materials.append("cotton")
            elif "linen" in details:
                materials.append("linen")
            elif "suede" in details:
                materials.append("suede")
            elif "leather" in details and "faux" in details:
                materials.append("faux leather")
            
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
    st.write("Search for fashion products using text, image, or both!")
    
    # Load data and models
    df, loader = load_data()
    text_embedding_generator, image_embedding_generator, vector_db = initialize_models()
    
    # Build or load indexes
    build_or_load_indexes(df, text_embedding_generator, image_embedding_generator, vector_db)
    
    # Create tabs for different search modes
    tab1, tab2, tab3, tab4 = st.tabs(["Text Search", "Image Search", "Hybrid Search", "Gemini Insights"])
    
    with tab1:
        st.header("Search by Text")
        text_query = st.text_input("Enter your search query", "leather jacket with pockets", key="text_search_query")
        
        if st.button("Search by Text"):
            if text_query.strip():
                with st.spinner("Searching..."):
                    # Generate text embedding for the query
                    query_embedding = text_embedding_generator.generate_text_embedding(text_query)
                    
                    # Search by text
                    distances, indices = vector_db.search_by_text(query_embedding, k=5)
                    
                    # Display results
                    st.subheader("Results")
                    
                    # Get product details
                    products = loader.get_product_details(indices[0].tolist())
                    
                    # Display RAG description
                    st.markdown(generate_rag_description(products, text_query))
                    
                    # Display product cards
                    for product in products:
                        st.divider()
                        display_product(product)
    
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
                    
                    # Display RAG description
                    st.markdown(generate_rag_description(products, "your image"))
                    
                    # Display product cards
                    for product in products:
                        st.divider()
                        display_product(product)
    
    with tab3:
        st.header("Hybrid Search (Text + Image)")
        
        hybrid_text = st.text_input("Enter your search query (hybrid)", "casual jacket", key="hybrid_text_query")
        
        # Option to upload an image
        hybrid_file = st.file_uploader("Choose an image (optional)", type=["jpg", "jpeg", "png"], key="hybrid_image_uploader")
        
        # Option to adjust weights
        text_weight = st.slider("Text search weight", 0.0, 1.0, 0.5, 0.1, key="text_weight_slider")
        image_weight = 1.0 - text_weight
        
        if st.button("Search with Hybrid Approach"):
            # Check if at least one search criterion is provided
            if hybrid_text.strip() or hybrid_file is not None:
                with st.spinner("Searching with hybrid approach..."):
                    text_embedding = None
                    image_embedding = None
                    
                    # Generate text embedding if text provided
                    if hybrid_text.strip():
                        text_embedding = text_embedding_generator.generate_text_embedding(hybrid_text)
                    
                    # Generate image embedding if image provided
                    if hybrid_file is not None:
                        image = Image.open(hybrid_file)
                        img_array = image_embedding_generator.preprocess_image(image)
                        image_embedding = image_embedding_generator.image_model.predict(img_array)[0]
                    
                    # Perform hybrid search
                    indices = vector_db.hybrid_search(
                        text_embedding=text_embedding,
                        image_embedding=image_embedding,
                        k=5,
                        alpha=text_weight
                    )
                    
                    # Display results
                    st.subheader("Results")
                    
                    # Get product details
                    products = loader.get_product_details(indices)
                    
                    # Display RAG description
                    st.markdown(generate_rag_description(products, hybrid_text if hybrid_text.strip() else "your criteria"))
                    
                    # Display product cards
                    for product in products:
                        st.divider()
                        display_product(product)
    
    with tab4:
        st.header("Gemini Multimodal Insights")
        st.write("Get AI-powered insights about fashion items using both text and images.")
        
        # Text input for query or instructions
        gemini_text = st.text_area(
            "Enter your question or instructions",
            value="What kind of jacket is this? Describe the style, materials, and suggest occasions where it would be appropriate to wear it.",
            height=100,
            key="gemini_text_area"
        )
        
        # Image upload
        gemini_file = st.file_uploader("Upload an image of a fashion item", type=["jpg", "jpeg", "png"], key="gemini_uploader")
        
        # Option to use a URL for the image
        gemini_image_url = st.text_input("Or enter an image URL", key="gemini_image_url")
        
        if st.button("Get Gemini Insights", key="gemini_insights_button"):
            image = None
            
            # Get image from file or URL
            if gemini_file is not None:
                image = Image.open(gemini_file)
                st.image(image, caption="Uploaded image", width=300)
            elif gemini_image_url.strip():
                image = download_image(gemini_image_url)
                if image:
                    st.image(image, caption="Image from URL", width=300)
            
            if gemini_text.strip() and image is not None:
                with st.spinner("Generating insights from Gemini..."):
                    # Use the multimodal model to generate insights
                    insights = text_embedding_generator.generate_multimodal_description(gemini_text, image)
                    
                    # Display the results
                    st.subheader("Gemini's Insights")
                    st.markdown(insights)
                    
                    # Prompt for product search
                    if st.button("Find similar products based on these insights", key="find_similar_button"):
                        # Extract key terms from insights for search
                        search_query = insights.split(".")[0]  # Use first sentence as query
                        
                        with st.spinner("Searching for similar products..."):
                            # Generate text embedding for the query
                            query_embedding = text_embedding_generator.generate_text_embedding(search_query)
                            
                            # Search by text
                            distances, indices = vector_db.search_by_text(query_embedding, k=5)
                            
                            # Get product details
                            products = loader.get_product_details(indices[0].tolist())
                            
                            # Display product cards
                            st.subheader("Similar Products")
                            for product in products:
                                st.divider()
                                display_product(product)
            elif not image:
                st.warning("Please upload an image or provide an image URL.")
            elif not gemini_text.strip():
                st.warning("Please enter a question or instructions for Gemini.")

if __name__ == "__main__":
    main()