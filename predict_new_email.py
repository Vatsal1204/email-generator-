"""
predict_new_email.py - Classify new emails with your trained model
"""

import torch
import pandas as pd
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import pickle

# Load model and label encoder
checkpoint = torch.load("ultra_fast_model.pt", weights_only=False)
label_encoder = checkpoint['label_encoder']

# Load model architecture
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(label_encoder.classes_)
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def predict_email(email_text):
    """Predict intent of a single email"""
    
    # Tokenize
    inputs = tokenizer(
        email_text,
        truncation=True,
        padding=True,
        max_length=64,
        return_tensors='pt'
    )
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)[0]
        prediction = torch.argmax(outputs.logits, dim=1).item()
    
    # Get results
    intent = label_encoder.classes_[prediction]
    confidence = probabilities[prediction].item()
    
    # Get all probabilities
    all_probs = {
        label: probabilities[i].item() 
        for i, label in enumerate(label_encoder.classes_)
    }
    
    return intent, confidence, all_probs

# Example usage
if __name__ == "__main__":
    print("="*60)
    print("EMAIL INTENT CLASSIFIER")
    print("="*60)
    
    while True:
        print("\n" + "-"*40)
        email = input("Enter email text (or 'quit' to exit):\n> ")
        
        if email.lower() == 'quit':
            break
        
        if not email.strip():
            continue
        
        intent, confidence, all_probs = predict_email(email)
        
        print(f"\n📧 PREDICTION: {intent.upper()}")
        print(f"🎯 Confidence: {confidence:.2%}")
        print("\n📊 All probabilities:")
        for label, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(prob * 50)
            print(f"  {label:12}: {prob:5.2%} {bar}")