import time
from pythonosc import udp_client, dispatcher, osc_server
from camera import capture_image
from model_llava import process_image
from utils import validate_transcription
from urban_legends import urban_legends
import threading
import signal
import torch
import gc

# Initialize the OSC client
client = udp_client.SimpleUDPClient("127.0.0.1", 9000)
client2 = udp_client.SimpleUDPClient("192.168.1.22", 9000)

# Initialize global state
current_urban_legend_index = 0
current_iteration = 0
max_iterations = 30  # Maximum number of iterations per urban_legend
max_time_per_urban_legend = 300  # 5 minutes in seconds
last_valid_transcription = ""
start_next_loop = threading.Event()
first_run = True
running = True
debug = True

def on_osc_message(address, *args):
    """Handler for incoming OSC messages to start the next loop."""
    print(f"Received OSC message: {address}, {args}")
    start_next_loop.set()

def setup_osc_server():
    """Set up the OSC server to listen for incoming messages."""
    osc_dispatcher = dispatcher.Dispatcher()
    osc_dispatcher.map("/start_next", on_osc_message)
    print("OSC server is set up to listen on /start_next")

    osc_server_instance = osc_server.ThreadingOSCUDPServer(("0.0.0.0", 9001), osc_dispatcher)
    server_thread = threading.Thread(target=osc_server_instance.serve_forever)
    server_thread.start()
    print("OSC server is running and listening on port 9001")
    return osc_server_instance, server_thread

def signal_handler(sig, frame):
    """Handle termination signals to gracefully shut down."""
    global running
    print('Terminating the process...')
    running = False
    start_next_loop.set()

def process_urban_legend(urban_legend, is_first_time):
    """Process a urban_legend of text and send it through OSC."""
    global last_valid_transcription
    if is_first_time:
        # Send the original urban_legend text without modification
        transcribed_text = urban_legend
    else:
        # Capture an image from the webcam
        image = capture_image()
        if image is None:
            return

        # Ensure memory is released after processing
        with torch.no_grad():
            initial_transcription = process_image(image)

        # Validate the transcription
        transcribed_text = validate_transcription(initial_transcription, last_valid_transcription)
        if transcribed_text != last_valid_transcription:
            last_valid_transcription = transcribed_text
        
        del image, initial_transcription
        torch.cuda.empty_cache()
        gc.collect()

    print(transcribed_text)
    client.send_message("/tael/text", transcribed_text)
    client2.send_message("/tael/text", transcribed_text)

    if debug:
        log_transcription(current_urban_legend_index, current_iteration, transcribed_text)

def log_transcription(urban_legend_index, iteration, text):
    """Log the transcription to a file if debug mode is enabled."""
    with open("transcription_log.txt", "a") as log_file:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_file.write(f"{timestamp} - urban legend {urban_legend_index} iteration {iteration} -> {text}\n")


def main_loop():
    global first_run, current_iteration, current_urban_legend_index

    osc_server_instance, server_thread = setup_osc_server()
    signal.signal(signal.SIGINT, signal_handler)

    start_time = time.time()

    while running:
        if current_iteration >= max_iterations or time.time() - start_time > max_time_per_urban_legend:
            current_urban_legend_index = (current_urban_legend_index + 1) % len(urban_legends)
            current_iteration = 0
            first_run = True  # Reset for the new urban_legend
            print(f"Moving to the next urban_legend: {urban_legends[current_urban_legend_index]}")

            #reset time on next urban legend
            start_time = time.time()

        if not first_run:
            print(f"Waiting for OSC message to start the next loop: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
            start_next_loop.wait()
            start_next_loop.clear()
            if not running:
                break

        first_run = False

        print(f"Processing urban_legend: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        process_urban_legend(urban_legends[current_urban_legend_index], current_iteration == 0)
        
        # Explicit memory cleanup after every iteration
        torch.cuda.empty_cache()
        gc.collect()

        current_iteration += 1
        print("urban legend index: ", current_urban_legend_index, "iteration: ", current_iteration, "time: ", time.time() - start_time)

    osc_server_instance.shutdown()
    server_thread.join()
    print("OSC server has been shut down.")

if __name__ == '__main__':
    main_loop()
