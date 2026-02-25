"""
prepare_dataloader_final.py - Prepare and SAVE datasets for training
"""

# Safe torch import
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

import pandas as pd
from transformers import RobertaTokenizer
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import pickle
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================
# CONFIGURATION
# ============================================
class Config:
    MODEL_NAME = "roberta-base"
    MAX_LENGTH = 256
    BATCH_SIZE = 16
    RANDOM_SEED = 42

# ============================================
# DATASET CLASS
# ============================================
class EmailDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# ============================================
# MAIN PREPARATION FUNCTION
# ============================================
def prepare_and_save_data():
    """Prepare datasets and save them for training"""
    
    logger.info("=" * 60)
    logger.info("PREPARING DATA FOR TRAINING")
    logger.info("=" * 60)
    
    # Create output directory
    os.makedirs("dataset_splits", exist_ok=True)
    
    # Load splits
    logger.info("Loading dataset splits...")
    train_df = pd.read_excel("dataset_splits/train.xlsx")
    val_df = pd.read_excel("dataset_splits/validation.xlsx")
    
    logger.info(f"Training samples: {len(train_df)}")
    logger.info(f"Validation samples: {len(val_df)}")
    
    # Extract texts and labels
    train_texts = train_df["email_text"].tolist()
    val_texts = val_df["email_text"].tolist()
    
    # Encode labels
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_df["label"])
    val_labels = label_encoder.transform(val_df["label"])
    
    # Save label mapping
    label_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
    logger.info(f"Label Mapping: {label_mapping}")
    logger.info(f"Number of classes: {len(label_mapping)}")
    
    # Save label encoder
    with open("dataset_splits/label_encoder.pkl", 'wb') as f:
        pickle.dump(label_encoder, f)
    logger.info("✓ Label encoder saved")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {Config.MODEL_NAME}")
    tokenizer = RobertaTokenizer.from_pretrained(Config.MODEL_NAME)
    
    # Save tokenizer
    tokenizer.save_pretrained("dataset_splits/tokenizer")
    logger.info("✓ Tokenizer saved")
    
    # Tokenize
    logger.info("Tokenizing training data...")
    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=Config.MAX_LENGTH,
        return_tensors=None
    )
    
    logger.info("Tokenizing validation data...")
    val_encodings = tokenizer(
        val_texts,
        truncation=True,
        padding=True,
        max_length=Config.MAX_LENGTH,
        return_tensors=None
    )
    
    # Create datasets
    train_dataset = EmailDataset(train_encodings, train_labels)
    val_dataset = EmailDataset(val_encodings, val_labels)
    
    # Save datasets
    logger.info("Saving datasets...")
    torch.save(train_dataset, "dataset_splits/train_dataset.pt")
    torch.save(val_dataset, "dataset_splits/val_dataset.pt")
    logger.info("✓ Datasets saved as .pt files")
    
    # Create data loaders (optional, for verification)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    
    # Verify saved files
    logger.info("\n" + "=" * 60)
    logger.info("VERIFYING SAVED FILES")
    logger.info("=" * 60)
    
    saved_files = os.listdir("dataset_splits")
    for file in saved_files:
        file_size = os.path.getsize(f"dataset_splits/{file}") / 1024  # KB
        logger.info(f"  {file:30} : {file_size:.2f} KB")
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ DATA PREPARATION COMPLETE")
    logger.info("=" * 60)
    logger.info("\nFiles saved in 'dataset_splits' folder:")
    logger.info("  - train.xlsx (original data)")
    logger.info("  - validation.xlsx (original data)")
    logger.info("  - train_dataset.pt (ready for training)")
    logger.info("  - val_dataset.pt (ready for validation)")
    logger.info("  - label_encoder.pkl")
    logger.info("  - tokenizer/ (folder with tokenizer files)")
    
    return train_dataset, val_dataset, label_encoder

# ============================================
# RUN PREPARATION
# ============================================
if __name__ == "__main__":
    try:
        train_dataset, val_dataset, label_encoder = prepare_and_save_data()
        
        # Test loading a sample batch
        logger.info("\n" + "=" * 60)
        logger.info("TESTING DATA LOADER")
        logger.info("=" * 60)
        
        test_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        sample_batch = next(iter(test_loader))
        
        logger.info(f"Sample batch shapes:")
        logger.info(f"  input_ids: {sample_batch['input_ids'].shape}")
        logger.info(f"  attention_mask: {sample_batch['attention_mask'].shape}")
        logger.info(f"  labels: {sample_batch['labels'].shape}")
        
        logger.info("\n✅ All good! You can now run train_model.py")
        
    except Exception as e:
        logger.error(f"Preparation failed: {e}")
        raise