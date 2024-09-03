import sys
import time
from pythonosc import udp_client, dispatcher, osc_server
from camera import Camera
from model_llava import process_image
from utils import validate_transcription
import threading
import signal
import torch
import gc

# Initialize the OSC client
client = udp_client.SimpleUDPClient("127.0.0.1", 9000)
client2 = udp_client.SimpleUDPClient("192.168.1.174", 9000)

last_valid_transcription = ""
start_next_loop = threading.Event()
inference_time = time.time()
first_run = True
running = True
debug = True
move_to_next_legend = False

def log_gpu_memory(total_runtime, timestamp, urban_legend_index, iteration, current_iteration):
    """Log GPU memory usage and other relevant information to a file."""
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    with open("gpu_memory_log.txt", "a") as f:
        f.write(
            f"Total Runtime: {total_runtime:.2f} seconds, "
            f"Timestamp: {timestamp}, "
            f"Urban Legend Index: {urban_legend_index}, "
            f"Urban Legend Iteration: {iteration}, "
            f"Total Iteration: {current_iteration}, "
            f"GPU Memory Allocated: {allocated / 1024 ** 2:.2f} MB, "
            f"GPU Memory Reserved: {reserved / 1024 ** 2:.2f} MB\n"
        )


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

def process_urban_legend(urban_legend, is_first_time, camera, urban_legend_index):
    global last_valid_transcription, move_to_next_legend

    start_time = time.time()

    if is_first_time:
        transcribed_text = urban_legend
    else:
        print(f"Taking image: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        capture_start_time = time.time()
        image = camera.capture_image()
        capture_end_time = time.time()
        print(f"Image captured: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} (Time taken: {capture_end_time - capture_start_time:.2f}s)")
        if image is None:
            return

        with torch.no_grad():
            print(f"Starting inference: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
            inference_start_time = time.time()
            initial_transcription = process_image(image)
            inference_end_time = time.time()
            print(f"Finished inference: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} (Time taken: {inference_end_time - inference_start_time:.2f}s)")

        transcribed_text = validate_transcription(initial_transcription, last_valid_transcription)

        if transcribed_text != last_valid_transcription:
            last_valid_transcription = transcribed_text
            print("text length:", len(transcribed_text.split()))
            if len(transcribed_text.split()) < 20:
                move_to_next_legend = True
        
        del image, initial_transcription
        torch.cuda.empty_cache()
        gc.collect()

    end_time = time.time()
    print(f"Fully finished: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} (Total time taken: {end_time - start_time:.2f}s)")

    print(transcribed_text)

    osc_address = f"/urban_legend/{urban_legend_index}"
    client.send_message(osc_address, transcribed_text)
    client2.send_message(osc_address, transcribed_text)

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

def main_loop(urban_legend, urban_legend_index):
    global first_run, current_iteration, start_next_loop, move_to_next_legend

    osc_server_instance, server_thread = setup_osc_server()
    signal.signal(signal.SIGINT, signal_handler)
    camera = Camera()

    start_time = time.time()
    current_iteration = 0
    max_iterations = 30  # Maximum number of iterations per urban_legend
    max_time_per_urban_legend = 300  # 5 minutes in seconds
    inference_time = time.time()
    iteration = 0  # Initialize iteration count

    while running:
        if current_iteration >= max_iterations or time.time() - start_time > max_time_per_urban_legend:
            print("Surpassed max iterations or time exceeded. Terminating...")
            break

        if not first_run and not move_to_next_legend:
            print(f"Waiting for OSC message to start the next loop: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
            start_next_loop.wait()
            start_next_loop.clear()
            if not running:
                break

        first_run = False
        print(f"Processing urban_legend: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        process_urban_legend(urban_legend, current_iteration == 0, camera, urban_legend_index)
        if(move_to_next_legend):
            print("Transcrip too short. Terminating...")
            break

        current_iteration += 1
        iteration += 1

        # Log GPU memory and other details
        total_runtime = time.time() - start_time
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        log_gpu_memory(total_runtime, timestamp, urban_legend_index, iteration, current_iteration)

        torch.cuda.empty_cache()
        gc.collect()
        

    osc_server_instance.shutdown()
    server_thread.join()
    camera.release()
    move_to_next_legend = False
    print("OSC server shut down and camera released")


if __name__ == '__main__':
    if len(sys.argv) > 2:
        urban_legend = sys.argv[1]  # Read the urban legend from the command-line argument
        urban_legend_index = int(sys.argv[2])
        main_loop(urban_legend, urban_legend_index)
    else:
        print("No urban legend provided. Exiting.")
