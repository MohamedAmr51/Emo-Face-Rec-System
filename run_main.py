import subprocess

subprocess.run([
    "start", "cmd", "/k", 
    "conda activate deepface_clone_emotion_rec_merge && python main.py "
], shell=True)