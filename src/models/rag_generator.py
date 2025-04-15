import numpy as np
from typing import List, Dict, Optional, Any, Tuple, Union
import requests
import json

class RAGGenerator:
    def __init__(self, vector_db, product_loader, api_key: str):
        self.vector_db = vector_db
        self.product_loader = product_loader
        self.api_key = api_key

    def extract_product_attributes(self, name: str, details: str) -> Tuple[List[str], List[str], List[str]]:
        categories = []
        materials = []
        features = []
        
        if details and isinstance(details, str):
            details_lower = details.lower()
            product_name_lower = name.lower() if name else ""
            
            # Categories
            if "bomber" in details_lower or "bomber" in product_name_lower:
                categories.append("bomber")
            elif "leather" in details_lower or "leather" in product_name_lower:
                categories.append("leather")
            elif "denim" in details_lower or "denim" in product_name_lower:
                categories.append("denim")
            elif "technical" in details_lower or "technical" in product_name_lower:
                categories.append("technical")
            elif "trench" in details_lower or "trench" in product_name_lower:
                categories.append("trench coat")
            
            # Materials
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
            elif "wool" in details_lower:
                materials.append("wool")
            
            # Features
            if "zip" in details_lower:
                features.append("zip closure")
            if "pocket" in details_lower:
                features.append("pockets")
            if "hood" in details_lower:
                features.append("hooded")
            if "elastic" in details_lower or "elasticated" in details_lower:
                features.append("elastic details")
            if "ribbed" in details_lower:
                features.append("ribbed trims")
            if "collar" in details_lower:
                if "lapel" in details_lower:
                    features.append("lapel collar")
                elif "high" in details_lower:
                    features.append("high neck")
                else:
                    features.append("collared")
                    
        return categories, materials, features

    def retrieve_similar_products(self, 
                                 query_embedding: np.ndarray, 
                                 search_type: str = 'text',
                                 k: int = 5,
                                 query_text: Optional[str] = None) -> List[Dict]:

        # Use the appropriate search method based on type
        if search_type == 'text':
            distances, indices = self.vector_db.search_by_text(
                query_embedding, k=k, query_text=query_text, keyword_boost=True)
        else:  # image search
            distances, indices = self.vector_db.search_by_image(query_embedding, k=k)
            
        # Get product details
        similar_products = self.product_loader.get_product_details(indices[0].tolist())
        return similar_products

    def build_similar_products_context(self, similar_products: List[Dict], target_product: Optional[Dict] = None) -> str:
        if not similar_products:
            return ""
        
        context_parts = []
        target_product_id = target_product.get("link", "") if target_product else ""
        
        for i, product in enumerate(similar_products):
            # Skip if this is the target product
            if target_product and product.get("link", "") == target_product_id:
                continue
                
            name = product.get("name", "")
            details = product.get("details", "")
            
            if name and details:
                context_parts.append(f"Similar Product {i+1}: {name}\nDetails: {details}")
        
        if not context_parts:
            return ""
            
        return "\n\n".join(context_parts)

    def build_llm_prompt_format(self) -> str:
        return (
            "Fill in the following format by writing only inside the brackets [] (but do not include the brackets in the output). Use bold font for key words. Follow the structure exactly.\n\n"
            "Format:\n"
            "Gemini Generated Description:\n\n"
            "[Write a creative, engaging product description of around 30-50 words. Include unique selling points and incorporate knowledge from similar products when relevant.]\n\n"
            "Key Features:\n"
            "- [Feature 1]\n"
            "- [Feature 2]\n"
            "- [Feature 3]\n\n"
            "Example:\n"
            "Gemini Generated Description:\n\n"
            "This ultra-soft hoodie blends comfort with street style—perfect for chilly evenings or laid-back weekends. Made from recycled fibers, it’s cozy, breathable, and eco-conscious.\n\n"
            "Key Features:\n"
            "- Made with 100% recycled materials\n"
            "- Unisex design with relaxed fit\n"
            "- Machine-washable and shrink-resistant\n"
        )

    def call_llm_api(self, messages: List[Dict], model: str = "google/gemini-2.0-flash-lite-001") -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://ai-product-recommender.app", 
            "X-Title": "AI Product Recommender"
        }
        
        payload = {
            "model": model,
            "messages": messages
        }
        
        try:
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, data=json.dumps(payload))
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        except requests.exceptions.RequestException as e:
            return f"Error during API call: {str(e)}"
        
        try:
            response_json = response.json()
            result_text = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
            return result_text if result_text else "No description generated."
        except (json.JSONDecodeError, IndexError, KeyError, AttributeError) as e:
            return f"Error parsing API response: {str(e)}"
        except Exception as e: # Catch any other unexpected errors during parsing
             return f"An unexpected error occurred while parsing the API response: {str(e)}"

    def generate_product_description_with_rag(self, 
                                             product: Dict,
                                             similar_products: List[Dict],
                                             query: Optional[str] = None) -> str:
        if not product:
            return ""
        
        name = product.get("name", "")
        details = product.get("details", "")

        categories, materials, features = self.extract_product_attributes(name, details)

        similar_context = self.build_similar_products_context(similar_products, product)

        # Build product info
        product_info = f"Product Name: {name}\nProduct Details: {details}"
        extra_info = ""
        if categories:
            extra_info += "Categories: " + ", ".join(categories) + "\n"
        if materials:
            extra_info += "Materials: " + ", ".join(materials) + "\n"
        if features:
            extra_info += "Features: " + ", ".join(features) + "\n"
        
        # Build prompt with RAG context
        prompt = ""
        if query:
            prompt += f"User Query: {query}\n\n"
        
        prompt += "Product Information:\n" + product_info + "\n\n"
        
        if extra_info:
            prompt += "Additional Details:\n" + extra_info + "\n\n"
            
        if similar_context:
            prompt += "Context from Similar Products:\n" + similar_context + "\n\n"

        prompt += self.build_llm_prompt_format()

        messages = [{"role": "user", "content": prompt}]
        return self.call_llm_api(messages=messages) # Use default text model

    def generate_image_description_with_rag(self,
                                           image,
                                           similar_products: List[Dict]) -> str:
        if not image:
            return ""
        
        # Convert image to bytes for API request
        import io
        import base64
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=image.format if image.format else 'JPEG')
        image_bytes = img_byte_arr.getvalue()
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')

        # Build context from similar products
        similar_context = self.build_similar_products_context(similar_products)

        # Build prompt
        prompt = (
            "You are an expert fashion analyzer. Describe this fashion item in detail.\n\n"
        )
        
        if similar_context:
            prompt += (
                f"Based on similar products in our database, this image might be related to:\n"
                f"{similar_context}\n\n"
                f"Use this context to enhance your description, but primarily focus on what you see in the image.\n\n"
            )

        prompt += self.build_llm_prompt_format()

        # Prepare the message payload for multimodal input
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                ]
            }
        ]

        # Call the unified API function with the appropriate model
        return self.call_llm_api(messages=messages, model="google/gemini-2.0-flash-001")
