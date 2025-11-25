import subprocess

subprocess.run([
    "start", "cmd", "/k", 
    "conda activate FIQ_gpu_env && python fixed_quality_score.py --data-dir ./data/quality_data --datasets custom_dataset --model_path ./CR-FIQAL --model_id 181952 --score_file_name custom_quality_scores.txt"
], shell=True)