"""
create_hf_repo.py - Create repository on Hugging Face
Run this first to create your repository
"""

from huggingface_hub import HfApi

# Your token
HF_TOKEN = "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"
YOUR_USERNAME = "vatsal124"
REPO_NAME = "email-classifier"

def create_repository():
    print("🚀 Creating repository on Hugging Face...")
    
    api = HfApi(token=HF_TOKEN)
    
    try:
        api.create_repo(
            repo_id=f"{YOUR_USERNAME}/{REPO_NAME}",
            repo_type="model",
            private=False,  # Set to True if you want private repo
            exist_ok=False  # Will fail if repo already exists
        )
        print(f"✅ Repository created successfully!")
        print(f"🌐 View it at: https://huggingface.co/{YOUR_USERNAME}/{REPO_NAME}")
    except Exception as e:
        print(f"❌ Failed to create repository: {e}")

if __name__ == "__main__":
    create_repository()