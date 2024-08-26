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

def on_osc_message(address, *args):
    """Handle incoming OSC messages."""
    print(f"Received OSC message: {address}, {args}")
    start_next_loop.set()  # Signal to start the next loop

def initialize_osc_clients():
    """Initialize the OSC clients."""
    global client, client2
    client = udp_client.SimpleUDPClient("127.0.0.1", 9000)
    client2 = udp_client.SimpleUDPClient("192.168.1.22", 9000)

def setup_osc_server():
    """Setup and start the OSC server."""
    global server, server_thread
    osc_dispatcher = dispatcher.Dispatcher()
    osc_dispatcher.map("/start_next", on_osc_message)
    print("OSC server is set up to listen on /start_next")

    server = osc_server.ThreadingOSCUDPServer(("0.0.0.0", 9001), osc_dispatcher)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.start()
    print("OSC server is running and listening on port 9001")

def wait_for_osc_signal():
    """Wait for an OSC signal to start the next loop."""
    print(f"Waiting for OSC message to start the next loop: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    start_next_loop.wait()
    start_next_loop.clear()

def capture_and_process_image():
    """Capture an image from the webcam and process it using the model."""
    global last_valid_transcription
    start_time = time.time()
    print(f"Taking picture: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

    image = capture_image()
    if image is None:
        return

    print(f"Starting inference: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    initial_transcription = process_image(image)
    print(f"Finished inference: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

    transcribed_text = validate_transcription(initial_transcription, last_valid_transcription)

    if transcribed_text != last_valid_transcription:
        last_valid_transcription = transcribed_text

    print(transcribed_text)
    return transcribed_text

def send_transcription_to_clients(transcribed_text):
    """Send the transcribed text to the OSC clients."""
    client.send_message("/tael/text", transcribed_text)
    client2.send_message("/tael/text", transcribed_text)

def signal_handler(sig, frame):
    """Handle SIGINT for graceful termination."""
    global running
    print('Terminating the process...')
    running = False
    start_next_loop.set()  # Unblock the wait call if blocked

def main():
    """Main function to run the loop."""
    global first_run, running

    initialize_osc_clients()
    setup_osc_server()
    signal.signal(signal.SIGINT, signal_handler)

    while running:
        if not first_run:
            wait_for_osc_signal()
            if not running:
                break

        first_run = False
        transcribed_text = capture_and_process_image()
        if transcribed_text:
            send_transcription_to_clients(transcribed_text)

    server.shutdown()
    server_thread.join()
    print("OSC server has been shut down.")

if __name__ == '__main__':
    main()
