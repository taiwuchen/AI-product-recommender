import os
import ssl
import requests
from typing import Optional
from dotenv import load_dotenv
from google.cloud import aiplatform
from google.oauth2 import service_account


# Fix SSL certificate verification on macOS
def fix_certificate_verification():
    """Fix SSL certificate verification issues on macOS"""
    try:
        # Use certifi if available
        import certifi
        os.environ['SSL_CERT_FILE'] = certifi.where()
        
        # For macOS, run the certificate install command for Python
        if os.path.exists('/Applications/Python 3.10/Install Certificates.command'):
            import subprocess
            subprocess.run(['/Applications/Python 3.10/Install Certificates.command'], check=False, shell=True)
        
        # Create unverified HTTPS context if needed
        ssl._create_default_https_context = ssl._create_unverified_context
    except Exception as e:
        print(f"Warning: Could not fix SSL certificate verification: {e}")


class BaseEmbeddingGenerator:
    """
    Base class for embedding generators with common functionality.
    """
    
    def __init__(self, google_credentials_path: Optional[str] = None, vertex_ai_region: Optional[str] = None):
        """
        Initialize the base embedding generator.
        
        Args:
            google_credentials_path (str, optional): Path to Google Cloud service account credentials.
            vertex_ai_region (str, optional): Google Cloud region for Vertex AI. If None, uses environment 
                                              variable VERTEX_AI_REGION or defaults to "us-central1".
        """
        load_dotenv()
        
        # Fix SSL certificate issues before making any requests
        fix_certificate_verification()
        
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
            print("Will use backup TensorFlow models for embeddings")