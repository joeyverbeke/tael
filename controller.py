import subprocess
import sys
import os
import time
from urban_legends import URBAN_LEGENDS

def run_script_in_virtualenv(script_name, args, venv_path="tael"):
    python_executable = os.path.join(venv_path, "Scripts", "python.exe") if os.name == "nt" else os.path.join(venv_path, "bin", "python")
    command = [python_executable, script_name] + args
    process = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr)
    return process

if __name__ == "__main__":
    urban_legend_index = 0
    restart_threshold = 10  # Restart every 10 iterations
    iteration_count = 0

    while True:
        current_urban_legend = URBAN_LEGENDS[urban_legend_index]
        print(f"Starting new urban legend: {current_urban_legend[:30]}...")

        # Run main.py with the current urban legend
        process = run_script_in_virtualenv("main.py", [current_urban_legend, str(urban_legend_index)])
        
        # Monitor the process and handle exceptions
        try:
            process.wait()  # Wait for the process to complete
        except subprocess.SubprocessError as e:
            print(f"Error occurred: {e}. Restarting script.")
            process.terminate()  # Terminate process on error
            process.wait()  # Ensure the process is terminated

        # Move to the next urban legend, loop back if at the end
        urban_legend_index = (urban_legend_index + 1) % len(URBAN_LEGENDS)
        iteration_count += 1

         # Periodically restart the Python environment to clear memory
        if iteration_count >= restart_threshold:
            print("Restarting entire Python environment to clear memory.")
            process.terminate()  # Terminate the process to free memory
            process.wait()  # Wait for termination to complete
            time.sleep(5)  # Short pause to allow for cleanup
            iteration_count = 0  # Reset iteration count after restart
            continue
