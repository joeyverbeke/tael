import subprocess
import sys
import os
import time
from urban_legends import URBAN_LEGENDS

def run_script_in_virtualenv(script_name, args, venv_path="tael"):
    """
    Run a Python script in a specified virtual environment.

    :param script_name: The name of the script to run.
    :param args: A list of arguments to pass to the script.
    :param venv_path: The path to the virtual environment (default is "tael").
    """
    # Construct the full path to the virtual environment's Python executable
    python_executable = os.path.join(venv_path, "Scripts", "python.exe") if os.name == "nt" else os.path.join(venv_path, "bin", "python")

    # Construct the command to run the script in the virtual environment
    command = [python_executable, script_name] + args

    # Run the command and capture the output
    process = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr)
    process.wait()

if __name__ == "__main__":
    urban_legend_index = 0  # Start with the first urban legend

    while True:
        # Get the current urban legend to process
        current_urban_legend = URBAN_LEGENDS[urban_legend_index]

        print(f"Starting new urban legend: {current_urban_legend[:30]}...")

        # Run main.py with the current urban legend
        run_script_in_virtualenv("main.py", [current_urban_legend])

        # Move to the next urban legend, loop back if at the end
        urban_legend_index = (urban_legend_index + 1) % len(URBAN_LEGENDS)
