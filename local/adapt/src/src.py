import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

def save_finetuned_model(base_model_id, adapter_path, output_dir):
    """
    Load base model and adapter, merge them, and save the result
    """
    print("Loading base model and tokenizer...")
    
    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True  # Added for Phi model
    )
    
    print("Loading adapter...")
    # Load and attach adapter with correct prefix
    config = PeftModel.from_pretrained(base_model, adapter_path).peft_config["default"]
    # Update the config to match Phi's architecture
    config.target_modules = ["self_attn.o_proj", "mlp.down_proj"]
    config.inference_mode = True
    
    # Create PeftModel with updated config
    model = PeftModel(base_model, config)
    # Load adapter weights
    model.load_adapter(adapter_path, adapter_name="default")
    
    print("Merging adapter with base model...")
    # Merge adapter with base model
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to {output_dir}...")
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save merged model and tokenizer
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("Model and tokenizer saved successfully!")

if __name__ == "__main__":
    base_model_id = "microsoft/Phi-3.5-mini-instruct"
    adapter_path = "../adapters/padme"
    output_dir = "../ft_models/ani"
    
    save_finetuned_model(base_model_id, adapter_path, output_dir)