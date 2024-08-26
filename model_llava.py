import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
from torch.cuda.amp import autocast  # Import for Automatic Mixed Precision

# Enable cuDNN auto-tuner to find the best algorithm for your hardware.
torch.backends.cudnn.benchmark = True

# Load the processor and model
model_id = "llava-hf/llava-v1.6-mistral-7b-hf"

processor = LlavaNextProcessor.from_pretrained(model_id)

model = LlavaNextForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,  # Use mixed precision for better performance
    low_cpu_mem_usage=True
)
model.to("cuda:0")  # Ensure the model is loaded onto the GPU

def process_image(image: Image.Image):
    """Generate a transcription from an image using the Llava model."""
    
    # Prepare the conversation prompt with [INST] tags
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "[INST] <image>\nPlease transcribe the text in this image. Your response should be in the format of ONLY the transcribed text, with no quotation marks or other prefacing. [/INST]"},
            ],
        },
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")

    # Define generation arguments
    generation_args = {
        "max_new_tokens": 200,  # Prevent cutting off
        "no_repeat_ngram_size": 3,  # Prevent repetition
        "length_penalty": 0.9,  # Balance between conciseness and completeness
        "eos_token_id": processor.tokenizer.eos_token_id,  # Correct end-of-sequence token
        "num_beams": 5,  # Beam search for better accuracy
        "temperature": 0.7,  # Lower temperature for more deterministic output
        "top_p": 0.9,  # Top-p sampling for controlled output diversity
        "top_k": 50    # Top-k sampling for focused predictions
    }

    # Use autocast for automatic mixed precision for better performance
    with torch.autocast("cuda", dtype=torch.float16):
        # Autoregressively complete the prompt
        output = model.generate(
            **inputs,
            **generation_args
        )

    # Decode and process the output
    response = processor.decode(output[0], skip_special_tokens=True)

    # Extract the transcribed text (after the prompt)
    transcribed_text = response.split("[/INST]")[-1].strip()
    
    return transcribed_text
