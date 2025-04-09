#!/usr/bin/env python
"""
Demo script for AI Product Recommendation System.
This script demonstrates the core functionality of the recommendation system.
"""

import os
import sys
import argparse
import pandas as pd
from PIL import Image
import requests
from io import BytesIO

# Add src to the path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from src.utils.data_loader import ProductDataLoader
from src.models.embeddings import EmbeddingGenerator, fix_certificate_verification
from src.models.vector_db import VectorDatabase


def setup_system(data_path, indexes_dir):
    """Set up the recommendation system."""
    print("Setting up the recommendation system...")
    
    # Fix SSL certificate issues at the start
    fix_certificate_verification()
    
    # Create index directory if it doesn't exist
    os.makedirs(indexes_dir, exist_ok=True)

    # Load and preprocess the data
    print("Loading product data...")
    loader = ProductDataLoader(data_path)
    df = loader.preprocess_data()
    print(f"Loaded {len(df)} products.")

    # Initialize embedding generator with us-west2 region
    print("Initializing embedding generator...")
    embedding_generator = EmbeddingGenerator(vertex_ai_region="us-west2")
    
    # Initialize vector database
    vector_db = VectorDatabase()

    # Check if indexes already exist
    if os.path.exists(os.path.join(indexes_dir, 'text_index.faiss')) and \
       os.path.exists(os.path.join(indexes_dir, 'image_index.faiss')):
        print("Loading existing indexes...")
        vector_db.load_indices(indexes_dir)
    else:
        print("Building indexes (this may take a while)...")
        # Generate text embeddings
        print("Generating text embeddings...")
        text_embeddings = embedding_generator.generate_batch_text_embeddings(df['text_for_embedding'].tolist())
        
        # Generate image embeddings
        print("Generating image embeddings...")
        image_embeddings = embedding_generator.generate_batch_image_embeddings(df['first_image_url'].tolist())
        
        # Add embeddings to vector db
        print("Adding embeddings to vector database...")
        vector_db.add_text_embeddings(text_embeddings, list(range(len(df))))
        vector_db.add_image_embeddings(image_embeddings, list(range(len(df))))
        
        # Save indexes
        print("Saving indexes...")
        vector_db.save_indices(indexes_dir)

    print("Setup complete!")
    return loader, embedding_generator, vector_db


def download_image(image_url):
    """Download image from URL."""
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None


def text_search(loader, embedding_generator, vector_db, query, k=5):
    """Perform text-based search."""
    print(f"Searching for: '{query}'")
    
    # Generate text embedding for the query
    query_embedding = embedding_generator.generate_text_embedding(query)
    
    # Search by text
    distances, indices = vector_db.search_by_text(query_embedding, k=k)
    
    # Get product details
    products = loader.get_product_details(indices[0].tolist())
    
    return products


def image_search(loader, embedding_generator, vector_db, image_path=None, image_url=None, k=5):
    """Perform image-based search."""
    image = None
    
    if image_path:
        print(f"Searching with image from: {image_path}")
        image = Image.open(image_path)
    elif image_url:
        print(f"Searching with image from URL: {image_url}")
        image = download_image(image_url)
    else:
        print("Error: No image provided")
        return []
    
    if image:
        # Preprocess image
        img_array = embedding_generator.preprocess_image(image)
        
        # Generate image embedding
        query_embedding = embedding_generator.image_model.predict(img_array)[0]
        
        # Search by image
        distances, indices = vector_db.search_by_image(query_embedding, k=k)
        
        # Get product details
        products = loader.get_product_details(indices[0].tolist())
        
        return products
    
    return []


def hybrid_search(loader, embedding_generator, vector_db, text_query=None, 
                  image_path=None, image_url=None, text_weight=0.5, k=5):
    """Perform hybrid search combining text and image."""
    text_embedding = None
    image_embedding = None
    
    # Generate text embedding if text provided
    if text_query:
        print(f"Text query: '{text_query}'")
        text_embedding = embedding_generator.generate_text_embedding(text_query)
    
    # Generate image embedding if image provided
    image = None
    if image_path:
        print(f"Image from: {image_path}")
        image = Image.open(image_path)
    elif image_url:
        print(f"Image from URL: {image_url}")
        image = download_image(image_url)
        
    if image:
        img_array = embedding_generator.preprocess_image(image)
        image_embedding = embedding_generator.image_model.predict(img_array)[0]
    
    # Perform hybrid search
    if text_embedding is not None or image_embedding is not None:
        indices = vector_db.hybrid_search(
            text_embedding=text_embedding,
            image_embedding=image_embedding,
            k=k,
            alpha=text_weight
        )
        
        # Get product details
        products = loader.get_product_details(indices)
        
        return products
    
    print("Error: No query provided")
    return []


def print_products(products):
    """Print product details."""
    print(f"\nFound {len(products)} matching products:\n")
    
    for i, product in enumerate(products):
        print(f"Product {i+1}: {product['name']}")
        print(f"Description: {product['details']}")
        print(f"Link: {product['link']}")
        print(f"Image: {product['image_url']}")
        print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description='AI Product Recommendation Demo')
    
    # Mode selection
    parser.add_argument('--mode', type=str, choices=['text', 'image', 'hybrid'], default='text',
                        help='Search mode: text, image, or hybrid')
    
    # Text search parameters
    parser.add_argument('--query', type=str, help='Text query for search')
    
    # Image search parameters
    parser.add_argument('--image_path', type=str, help='Path to an image file')
    parser.add_argument('--image_url', type=str, help='URL to an image')
    
    # Hybrid search parameters
    parser.add_argument('--text_weight', type=float, default=0.5, 
                        help='Weight for text search in hybrid mode (0.0-1.0)')
    
    # General parameters
    parser.add_argument('--k', type=int, default=5, help='Number of results to return')
    
    args = parser.parse_args()
    
    # Configure paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'ZARA_jackets_men.csv')
    indexes_dir = os.path.join(current_dir, 'indexes')
    
    # Set up system
    loader, embedding_generator, vector_db = setup_system(data_path, indexes_dir)
    
    # Perform search based on the selected mode
    if args.mode == 'text':
        if not args.query:
            print("Error: Text query is required for text search")
            return
        products = text_search(loader, embedding_generator, vector_db, args.query, k=args.k)
    
    elif args.mode == 'image':
        if not args.image_path and not args.image_url:
            print("Error: Image path or URL is required for image search")
            return
        products = image_search(loader, embedding_generator, vector_db,
                                image_path=args.image_path, image_url=args.image_url, k=args.k)
    
    elif args.mode == 'hybrid':
        if not args.query and not args.image_path and not args.image_url:
            print("Error: Text query or image is required for hybrid search")
            return
        products = hybrid_search(loader, embedding_generator, vector_db, 
                                text_query=args.query, image_path=args.image_path, 
                                image_url=args.image_url, text_weight=args.text_weight, k=args.k)
    
    # Print results
    print_products(products)


if __name__ == "__main__":
    main()