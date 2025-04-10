import argparse
import numpy as np
from typing import List
from models.text_embedding import TextEmbeddingGenerator


def compute_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        embedding1 (np.ndarray): First embedding vector.
        embedding2 (np.ndarray): Second embedding vector.
        
    Returns:
        float: Cosine similarity score between 0 and 1.
    """
    # Ensure embeddings are normalized
    embedding1 = embedding1 / np.linalg.norm(embedding1)
    embedding2 = embedding2 / np.linalg.norm(embedding2)
    
    # Compute cosine similarity
    return float(np.dot(embedding1, embedding2))


def main():
    """
    Demo script to showcase text embeddings and similarity search.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Demo for text embeddings")
    parser.add_argument("--query", type=str, default="blue denim jacket with buttons",
                        help="Text query to embed")
    parser.add_argument("--google_credentials_path", type=str, default=None,
                        help="Path to Google Cloud credentials file (optional)")
    args = parser.parse_args()
    
    # Create text embedding generator
    try:
        text_embedder = TextEmbeddingGenerator(google_credentials_path=args.google_credentials_path)
        
        # Generate embedding for the query
        print(f"\nGenerating embedding for query: '{args.query}'")
        query_embedding = text_embedder.generate_text_embedding(args.query)
        print(f"Query embedding shape: {query_embedding.shape}")
        print(f"Query embedding sample: {query_embedding[:5]}...")
        
        # Define some product descriptions to compare against
        products = [
            "Blue denim jacket with silver buttons and collar",
            "Red leather jacket with zipper closure",
            "White cotton t-shirt with logo print",
            "Black skinny jeans with ripped knees",
            "Navy blue blazer with gold buttons",
            "Green military jacket with hood",
            "Striped button-up shirt with long sleeves",
            "Grey hoodie with front pocket"
        ]
        
        # Generate embeddings for all products
        print("\nGenerating embeddings for product descriptions...")
        product_embeddings = text_embedder.generate_batch_text_embeddings(products)
        print(f"Generated {len(product_embeddings)} product embeddings")
        
        # Compute similarities between query and products
        print("\nComputing similarities between query and products:")
        similarities = []
        for i, product in enumerate(products):
            similarity = compute_cosine_similarity(query_embedding, product_embeddings[i])
            similarities.append((product, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Print results
        print("\nResults sorted by similarity to query:")
        for i, (product, similarity) in enumerate(similarities):
            print(f"{i+1}. {product} - Similarity: {similarity:.4f}")
    
    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: This demo requires Google Cloud credentials for Vertex AI.")
        print("Please set up Google Cloud credentials and try again.")


if __name__ == "__main__":
    main() 