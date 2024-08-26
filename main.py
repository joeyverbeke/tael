import time
from pythonosc import udp_client, dispatcher, osc_server
from camera import capture_image
from model_llava import process_image
from utils import validate_transcription
import threading
import signal

# Initialize the OSC client
client = udp_client.SimpleUDPClient("127.0.0.1", 9000)
client2 = udp_client.SimpleUDPClient("192.168.1.22", 9000)

# Initialize conversation history
last_valid_transcription = ""

# Flag to control the start of the loop
start_next_loop = threading.Event()
first_run = True  # Flag to indicate the first run
running = True  # Flag to control the main loop

# Function to handle incoming OSC messages
def on_osc_message(address, *args):
    print(f"Received OSC message: {address}, {args}")
    start_next_loop.set()  # Signal to start the next loop

# Setup OSC server
dispatcher = dispatcher.Dispatcher()
dispatcher.map("/start_next", on_osc_message)  # TouchDesigner should send this message
print("OSC server is set up to listen on /start_next")

server = osc_server.ThreadingOSCUDPServer(("0.0.0.0", 9001), dispatcher)
server_thread = threading.Thread(target=server.serve_forever)
server_thread.start()
print("OSC server is running and listening on port 9001")

# Signal handler for graceful termination
def signal_handler(sig, frame):
    global running
    print('Terminating the process...')
    running = False
    start_next_loop.set()  # Unblock the wait call if blocked

signal.signal(signal.SIGINT, signal_handler)

while running:
    if not first_run:
        print(f"Waiting for OSC message to start the next loop: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        # Wait indefinitely until an OSC message is received or the process is terminated
        start_next_loop.wait()
        start_next_loop.clear()  # Reset the event after it has been set

        # Check if the process should terminate
        if not running:
            break

    # Reset first_run after the first iteration
    first_run = False

    start_time = time.time()
    print(f"Taking picture: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

    # Capture an image from the webcam
    image = capture_image()
    if image is None:
        continue

    print(f"Starting inference: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    
    # Process the image with the model
    initial_transcription = process_image(image)

    print(f"Finished inference: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

    # Use the validate_transcription function to get the valid transcription
    transcribed_text = validate_transcription(initial_transcription, last_valid_transcription)

    # Update the last valid transcription if the new one is valid
    if transcribed_text != last_valid_transcription:
        last_valid_transcription = transcribed_text

    # Print the transcribed text
    print(transcribed_text)

    # Send the transcribed text to the OSC clients
    client.send_message("/tael/text", transcribed_text)
    client2.send_message("/tael/text", transcribed_text)

# Gracefully shutdown the OSC server
server.shutdown()
server_thread.join()
print("OSC server has been shut down.")
