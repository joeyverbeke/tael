import subprocess
import sys
import os
from urban_legends import URBAN_LEGENDS

def run_script_in_virtualenv(script_name, args, venv_path="tael"):
    python_executable = os.path.join(venv_path, "Scripts", "python.exe") if os.name == "nt" else os.path.join(venv_path, "bin", "python")
    command = [python_executable, script_name] + args
    process = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr)
    return process

if __name__ == "__main__":
    urban_legend_index = 0
    restart_threshold = 10  # Restart every 10 iterations

    while True:
        current_urban_legend = URBAN_LEGENDS[urban_legend_index]
        print(f"Starting new urban legend: {current_urban_legend[:30]}...")

        # Run main.py with the current urban legend
        process = run_script_in_virtualenv("main.py", [current_urban_legend])
        process.wait()

        urban_legend_index = (urban_legend_index + 1) % len(URBAN_LEGENDS)

        # Periodically restart the process to clear memory
        if urban_legend_index % restart_threshold == 0:
            print("Restarting process to clear memory...")
            process.terminate() 
            process.wait()
            continue
