import time
from pythonosc import udp_client
from camera import capture_image
from model_llava import process_image
from utils import validate_transcription

# Initialize the OSC client
client = udp_client.SimpleUDPClient("127.0.0.1", 9000)

# Initialize conversation history
last_valid_transcription = ""

while True:
    start_time = time.time()

    # Capture an image from the webcam
    image = capture_image()
    if image is None:
        continue

    # Process the image with the model
    response = process_image(image)

    # Validate the transcription
    if validate_transcription(response):
        last_valid_transcription = response
    else:
        response = last_valid_transcription

    # Print the transcribed text
    print(response)

    # Send the transcribed text to the OSC client
    client.send_message("/tael/text", response)

    # Calculate elapsed time and wait if necessary
    elapsed_time = time.time() - start_time
    if elapsed_time < 30:
        time.sleep(30 - elapsed_time)
