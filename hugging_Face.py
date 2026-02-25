"""
upload_to_huggingface.py - Upload your model to Hugging Face
Run this script to upload your 256MB model file
"""

from huggingface_hub import HfApi
import os

# ============================================
# CONFIGURATION - USE YOUR TOKEN
# ============================================
HF_TOKEN = "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"  # Your token
YOUR_USERNAME = "vatsal124"  # Your Hugging Face username
REPO_NAME = "email-classifier"  # Repository name
MODEL_FILE = "ultra_fast_model.pt"  # Your model file

# ============================================
# UPLOAD THE MODEL
# ============================================
def upload_model():
    print("🚀 Starting upload to Hugging Face...")
    
    # Initialize API with your token
    api = HfApi(token=HF_TOKEN)
    
    # Full repo ID
    repo_id = f"{YOUR_USERNAME}/{REPO_NAME}"
    
    # Check if file exists
    if not os.path.exists(MODEL_FILE):
        print(f"❌ Error: {MODEL_FILE} not found in current folder!")
        print(f"Current folder: {os.getcwd()}")
        return
    
    file_size = os.path.getsize(MODEL_FILE) / (1024 * 1024)
    print(f"📁 Found model file: {MODEL_FILE} ({file_size:.1f} MB)")
    
    # Upload the file
    print("📤 Uploading to Hugging Face... (this may take a few minutes)")
    
    try:
        api.upload_file(
            path_or_fileobj=MODEL_FILE,
            path_in_repo=MODEL_FILE,
            repo_id=repo_id,
            repo_type="model",
        )
        
        print(f"✅ Upload successful!")
        print(f"🌐 Your model is now at: https://huggingface.co/{repo_id}")
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        print("\nPossible issues:")
        print("1. Repository doesn't exist - create it first")
        print("2. Token doesn't have write permissions")
        print("3. File path is incorrect")

if __name__ == "__main__":
    upload_model()