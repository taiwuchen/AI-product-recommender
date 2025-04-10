import argparse
import numpy as np
from models.image_embedding import ImageEmbeddingGenerator


def main():
    """
    Demo script to showcase CLIP image and text embeddings.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Demo for CLIP image and text embeddings")
    parser.add_argument("--image_url", type=str, default="https://images.unsplash.com/photo-1591047139829-d91aecb6caea?ixlib=rb-4.0.3",
                        help="URL of the image to embed")
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32",
                       help="Name of the CLIP model to use")
    parser.add_argument("--text_query", type=str, default="a photo of a jacket",
                       help="Text query to compare with the image")
    args = parser.parse_args()
    
    # Create CLIP embedding generator
    clip_embedder = ImageEmbeddingGenerator(
        clip_model_name=args.model_name
    )
    
    # Generate image embedding
    print(f"\nGenerating embedding for image: {args.image_url}")
    image_embedding = clip_embedder.generate_image_embedding(args.image_url)
    print(f"Image embedding shape: {image_embedding.shape}")
    print(f"Image embedding sample: {image_embedding[:5]}...")
    
    # Generate text embedding
    print(f"\nGenerating embedding for text: '{args.text_query}'")
    text_embedding = clip_embedder.generate_text_embedding(args.text_query)
    print(f"Text embedding shape: {text_embedding.shape}")
    print(f"Text embedding sample: {text_embedding[:5]}...")
    
    # Compare image and text embeddings
    similarity = clip_embedder.compute_similarity(image_embedding, text_embedding)
    print(f"\nSimilarity between image and text '{args.text_query}': {similarity:.4f}")
    
    # Try different text queries to compare
    comparison_texts = [
        "a photo of a chair",
        "a photo of a table",
        "a photo of a dog",
        "a photo of furniture",
        "a photo of outdoor scenery",
        "a photo of a jacket",
        "a photo of clothing"
    ]
    
    print("\nComparing image with different text queries:")
    for text in comparison_texts:
        text_emb = clip_embedder.generate_text_embedding(text)
        sim = clip_embedder.compute_similarity(image_embedding, text_emb)
        print(f"Similarity with '{text}': {sim:.4f}")


if __name__ == "__main__":
    main() 