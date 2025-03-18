import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def download_model():
    """
    Download and save the Qwen2.5-VL-3B-Instruct model and tokenizer
    """
    print("Loading configuration...")
    with open('../dataset/config.json', 'r') as f:
        config = json.load(f)
    
    model_name = config['model_config']['model_name']
    save_path = os.path.join('checkpoints', 'pretrained')
    
    print(f"Downloading model: {model_name}")
    print(f"This may take a while...")
    
    # Create save directory
    os.makedirs(save_path, exist_ok=True)
    
    try:
        # Download tokenizer
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(save_path)
        print("Tokenizer downloaded successfully!")
        
        # Download model
        print("Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map='auto'
        )
        model.save_pretrained(save_path)
        print("Model downloaded successfully!")
        
        print(f"\nModel and tokenizer saved to: {save_path}")
        
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        raise

if __name__ == "__main__":
    download_model() 