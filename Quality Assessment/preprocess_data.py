from utils.serfiq_example import align_faces_cpu , SendMail
from utils.MTCNN_alignment_fast import align_faces_gpu
import subprocess



faces_dir = "data\\Close-up Images\\person_3"
in_folder = "data\\Failed detected images"      # path to input images
out_folder = r"data\\aligned_for_failed"        # path to save aligned images
gpu = 0

mail_text = "preprocessing and facial alignment finished sucessfully !"
mail_subject = "Close up images Preprocessing"

align_faces_cpu(faces_dir)
align_faces_gpu(in_folder,out_folder,gpu)
subprocess.run(["python", "fixed_extract.py"])
subprocess.run([
    "python", "fixed_quality_score.py",
    "--data-dir", "./data/quality_data",
    "--datasets", "custom_dataset",
    "--model_path", "./CR-FIQAL",
    "--model_id", "181952",
    "--score_file_name", "custom_quality_scores.txt"
])

# SendMail(mail_text,mail_subject)