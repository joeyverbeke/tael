import sys
import time
from pythonosc import udp_client, dispatcher, osc_server
from camera import capture_image
from model_llava import process_image
from utils import validate_transcription
import threading
import signal
import torch
import gc

# Initialize the OSC client
client = udp_client.SimpleUDPClient("127.0.0.1", 9000)
client2 = udp_client.SimpleUDPClient("192.168.1.22", 9000)

last_valid_transcription = ""
start_next_loop = threading.Event()
inference_time = 0
first_run = True
running = True
debug = True

def on_osc_message(address, *args):
    print(f"Received OSC message: {address}, {args}")
    start_next_loop.set()

def setup_osc_server():
    osc_dispatcher = dispatcher.Dispatcher()
    osc_dispatcher.map("/start_next", on_osc_message)
    print("OSC server is set up to listen on /start_next")

    osc_server_instance = osc_server.ThreadingOSCUDPServer(("0.0.0.0", 9001), osc_dispatcher)
    server_thread = threading.Thread(target=osc_server_instance.serve_forever)
    server_thread.start()
    print("OSC server is running and listening on port 9001")
    return osc_server_instance, server_thread

def signal_handler(sig, frame):
    global running
    print('Terminating the process...')
    running = False
    start_next_loop.set()

def process_urban_legend(urban_legend, is_first_time):
    global last_valid_transcription
    if is_first_time:
        transcribed_text = urban_legend
    else:
        image = capture_image()
        if image is None:
            return

        with torch.no_grad():
            initial_transcription = process_image(image)

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
        log_transcription(transcribed_text)
    

def log_transcription(text):
    with open("transcription_log.txt", "a") as log_file:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        global inference_time
        current_time = time.time()
        inference_time = current_time - inference_time
        log_file.write(f"{timestamp} | Inference: {inference_time}s | {text}\n")
        inference_time = current_time

def main_loop(urban_legend):
    global first_run, current_iteration, start_next_loop

    osc_server_instance, server_thread = setup_osc_server()
    signal.signal(signal.SIGINT, signal_handler)

    start_time = time.time()
    current_iteration = 0
    max_iterations = 30  # Maximum number of iterations per urban_legend
    max_time_per_urban_legend = 300  # 5 minutes in seconds
    inference_time = time.time()

    while running:
        if current_iteration >= max_iterations or time.time() - start_time > max_time_per_urban_legend:
            print("Max iterations or time exceeded. Terminating...")
            break

        if not first_run:
            print(f"Waiting for OSC message to start the next loop: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
            start_next_loop.wait()
            start_next_loop.clear()
            if not running:
                break

        first_run = False
        print(f"Processing urban_legend: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        process_urban_legend(urban_legend, current_iteration == 0)
        current_iteration += 1

        torch.cuda.empty_cache()
        gc.collect()
        

    osc_server_instance.shutdown()
    server_thread.join()
    print("OSC server has been shut down.")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        urban_legend = sys.argv[1]  # Read the urban legend from the command-line argument
        main_loop(urban_legend)
    else:
        print("No urban legend provided. Exiting.")
