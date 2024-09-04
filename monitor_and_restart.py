import os
import time
import subprocess
import sys
import psutil

# Configuration
LOG_FILE = "transcription_log.txt"
MAX_IDLE_TIME = 60  # 60 seconds
VENV_PATH = "tael"
PYTHON_SCRIPT = "controller.py"
INITIAL_DELAY = 60  # Initial delay to allow the system to stabilize (in seconds)
CHECK_INTERVAL = 5  # Check interval in seconds

def run_script_in_virtualenv(venv_path, script_name):
    """
    Run a Python script in a specified virtual environment.
    """
    python_executable = os.path.join(venv_path, "Scripts", "python.exe") if os.name == "nt" else os.path.join(venv_path, "bin", "python")
    command = [python_executable, script_name]
    process = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr)
    return process

def get_last_modified_time(file_path):
    """
    Get the last modified time of a file.
    """
    try:
        return os.path.getmtime(file_path)
    except OSError:
        return None

def kill_processes_by_script_name(script_name):
    """
    Kill all processes that are running a specific script.
    """
    for proc in psutil.process_iter(['pid', 'cmdline']):
        if proc.info['cmdline'] and script_name in proc.info['cmdline']:
            try:
                proc.kill()
                print(f"Terminated process {proc.info['cmdline']} with PID {proc.info['pid']}.")
            except psutil.NoSuchProcess:
                print(f"Process {proc.info['pid']} no longer exists.")
            except Exception as e:
                print(f"Failed to terminate process {proc.info['cmdline']} with PID {proc.info['pid']}: {e}")

def monitor_log_file():
    """
    Monitor the log file and restart the system if it hasn't been updated.
    """
    # Start the controller.py script initially
    print("Starting the controller...")
    process = run_script_in_virtualenv(VENV_PATH, PYTHON_SCRIPT)

    # Initial startup delay
    print(f"Initial delay of {INITIAL_DELAY} seconds...")
    time.sleep(INITIAL_DELAY)

    while True:
        last_modified_time = get_last_modified_time(LOG_FILE)
        
        if last_modified_time:
            time_since_last_update = time.time() - last_modified_time
            
            if time_since_last_update > MAX_IDLE_TIME:
                print(f"No update in {MAX_IDLE_TIME} seconds. Restarting the system...")
                
                # Terminate all instances of controller.py and main.py
                kill_processes_by_script_name("controller.py")
                kill_processes_by_script_name("main.py")

                # Restart the script
                process = run_script_in_virtualenv(VENV_PATH, PYTHON_SCRIPT)
                
                # Delay after restarting to allow the new process to initialize
                time.sleep(INITIAL_DELAY)
        
        time.sleep(CHECK_INTERVAL)  # Check every CHECK_INTERVAL seconds

if __name__ == "__main__":
    monitor_log_file()
