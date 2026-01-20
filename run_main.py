"""
Launcher Script for Main Application.

This script acts as a wrapper to launch the main application (`main.py`) inside
a specific Conda environment (`deepface_emotion_env`).

It uses `subprocess` to spawn a new command prompt window, ensuring the environment
is activated correctly before execution.
"""
import subprocess

subprocess.run([
    "start", "cmd", "/k", 
    "conda activate deepface_emotion_env && python main.py "
], shell=True)