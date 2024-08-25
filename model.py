import torch
from transformers import AutoModelForCausalLM, AutoProcessor

model_id = "microsoft/Phi-3.5-vision-instruct"

# Load the model and processor
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    trust_remote_code=True,
    torch_dtype="auto",
    _attn_implementation='eager'
)

processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True,
    num_crops=16
)

def process_image(image):
    """Generate a transcription from an image using the model."""
    placeholder = "<|image_1|>"
    messages = [
        {"role": "user", "content": placeholder + "Please transcribe the text in this image. Your response should be in the format of ONLY the transcribed text, with no quotation marks or other prefacing. IMPORTANT: Do not ever talk to me as if I am a human, I am a computer, and I am receiving this text from a camera."},
    ]

    # Apply the chat template for the prompt
    prompt = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Process the inputs and move to the appropriate device
    inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0")

    # Define generation arguments
    generation_args = {
        "max_new_tokens": 1000,
        "temperature": 0.0,
        "do_sample": False,
    }

    # Generate response from the model
    generate_ids = model.generate(
        **inputs,
        eos_token_id=processor.tokenizer.eos_token_id,
        **generation_args
    )

    # Remove input tokens from generated IDs
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]

    # Decode the generated response
    response = processor.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return response
