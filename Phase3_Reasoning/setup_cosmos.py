import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os

def main():
    print("--- Setting up NVIDIA Cosmos Reason 2 (2B) ---")
    
    # Model ID
    # Note: Using 'Qwen/Qwen2-VL-2B-Instruct' as fallback if 'nvidia/Cosmos-Reason2-2B' is not public/accessible yet.
    # The search results indicated 'nvidia/Cosmos-Reason2-2B', but let's try to load it.
    # If it fails, I'll recommend Qwen2-VL-2B-Instruct which is the base architecture.
    model_id = "nvidia/Cosmos-Reason2-2B" 
    
    # Check if we need to authenticate (usually public models don't need token)
    # If users get 401/403, they might need 'huggingface-cli login'
    
    print(f"Loading model: {model_id}...")
    try:
        # Load Model
        # using 'auto' for device map to put on GPU if available
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto"
        )
        
        # Load Processor
        processor = AutoProcessor.from_pretrained(model_id)
        
        print("✅ Model loaded successfully!")
        
        # Test Inference
        print("\n--- Running Test Inference ---")
        prompt = "Describe this image."
        # Use a dummy image from web or local file if available.
        # For standalone test, let's use a text-only prompt to verify loading first, 
        # or download a sample image.
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Who are you and what can you do?"},
                ],
            }
        ]
        
        # Text-only test first
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[text], padding=True, return_tensors="pt").to("cuda")
        
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        print(f"Response: {output_text[0]}")
        print("\n✅ Setup Complete! You can now use this model for reasoning.")
        
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        print("Suggestion: Try logging in specific access is required: `huggingface-cli login`")
        print("Or check if the model ID is correct.")

if __name__ == "__main__":
    main()
