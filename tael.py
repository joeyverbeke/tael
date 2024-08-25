from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import cv2
import time
from pythonosc import udp_client

# Initialize the OSC client
client = udp_client.SimpleUDPClient("127.0.0.1", 9000)

# Load the processor and model
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf", 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True
)
model.to("cuda:0")

# Initialize conversation history
last_valid_transcription = ""

# Function to validate the transcribed text
def validate_transcription(text):
    invalid_phrases = [
        "too blurry", 
        "Setting `pad_token_id`", 
        "no text visible", 
        "error",
        "does not contain",
        "too dark",
        "cannot transcribe",
        "the text in the image",
        "the image",
    ]
    return not any(phrase in text.lower() for phrase in invalid_phrases)

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 is usually the default camera

while True:
    start_time = time.time()

    # Capture an image from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image. Retrying...")
        continue

    # Convert the image to PIL format
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Save the image (optional)
    image.save("image.jpg")

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

    # Autoregressively complete the prompt
    output = model.generate(
        **inputs,
        max_new_tokens=200,  # Increased to prevent cutting off
        no_repeat_ngram_size=3,  # Prevent repetition
        length_penalty=0.9,  # Balance between conciseness and completeness
        eos_token_id=processor.tokenizer.eos_token_id,  # Correct end-of-sequence token
        num_beams=5,  # Use beam search for better accuracy
        temperature=0.7,  # Lower temperature for more deterministic output
        top_p=0.9,  # Top-p sampling for controlled output diversity
        top_k=50    # Top-k sampling for focused predictions
    )

    # Decode and process the output
    response = processor.decode(output[0], skip_special_tokens=True)

    # Extract the transcribed text (after the prompt)
    transcribed_text = response.split("[/INST]")[-1].strip()

    # Validate the transcription
    if validate_transcription(transcribed_text):
        last_valid_transcription = transcribed_text
    else:
        transcribed_text = last_valid_transcription

    # Print the transcribed text
    print(transcribed_text)

    # Send the transcribed text to the OSC client
    client.send_message("/tael/text", transcribed_text)

    # Calculate elapsed time and wait if necessary
    elapsed_time = time.time() - start_time
    if elapsed_time < 30:
        time.sleep(30 - elapsed_time)

# Release the webcam when done
cap.release()
cv2.destroyAllWindows()
