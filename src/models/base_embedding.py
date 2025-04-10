import os
from typing import Optional
from dotenv import load_dotenv
from google.cloud import aiplatform
from google.oauth2 import service_account

class BaseEmbeddingGenerator:
    
    def __init__(self, google_credentials_path: Optional[str] = None, vertex_ai_region: Optional[str] = None):
        load_dotenv()
        
        # Initialize Google Cloud credentials
        self.credentials = None
        self.initialized = False
        
        # Get region from parameter, environment variable, or use default
        # us-central1 is chosen as default since it typically has all services available
        self.vertex_ai_region = vertex_ai_region or os.environ.get("VERTEX_AI_REGION", "us-central1")
        
        try:
            if google_credentials_path and os.path.exists(google_credentials_path):
                self.credentials = service_account.Credentials.from_service_account_file(google_credentials_path)
                aiplatform.init(credentials=self.credentials, project=os.environ.get("GOOGLE_CLOUD_PROJECT"), 
                               location=self.vertex_ai_region)
                self.initialized = True
            else:
                # Use default credentials if path not provided or file doesn't exist
                aiplatform.init(location=self.vertex_ai_region)
                self.initialized = True
                
            print(f"Google Cloud Vertex AI initialized successfully with region: {self.vertex_ai_region}")
        except Exception as e:
            print(f"Warning: Google Cloud initialization failed: {e}")