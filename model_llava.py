import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image

torch.backends.cudnn.benchmark = True

model_id = "llava-hf/llava-v1.6-mistral-7b-hf"

processor = LlavaNextProcessor.from_pretrained(model_id)

model = LlavaNextForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,  # Use mixed precision for better performance
    low_cpu_mem_usage=True
)
model.to("cuda:0") 

def process_image(image: Image.Image):
    """Generate a transcription from an image using the Llava model."""
    with torch.no_grad():
        
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
            "num_beams": 3,  # Beam search for better accuracy
            "do_sample": False
        }
        
        with torch.autocast("cuda", dtype=torch.float16):
            output = model.generate(
                **inputs,
                **generation_args
            )

        response = processor.decode(output[0], skip_special_tokens=True)

        transcribed_text = response.split("[/INST]")[-1].strip()
        
        return transcribed_text
