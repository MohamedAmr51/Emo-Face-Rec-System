import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import shutil 

from utils.align_trans import norm_crop

from facenet_pytorch import MTCNN

def align_images(in_folder, out_folder, gpu):  
    mtcnn = MTCNN(select_largest=True, post_process=False, device=gpu)
    # mtcnn = MTCNN(
    # select_largest=True,
    # post_process=False,
    # device=gpu,
    # min_face_size=5,         # Detect smaller faces
    # thresholds=[0.4, 0.5, 0.6],  # More lenient thresholds
    # factor=0.5,
    # keep_all=False,
    # margin=0,
    # selection_method="largest_over_threshold"
    # )
    os.makedirs(out_folder, exist_ok=True)
    skipped_imgs = []
    
    identity_names = os.listdir(in_folder)
    for identity in tqdm(identity_names):
        os.makedirs(os.path.join(out_folder, identity), exist_ok=True)
        img_names = os.listdir(os.path.join(in_folder, identity))

        identity_path = os.path.join(in_folder,identity)
        total_length = len(os.listdir(identity_path))

        with tqdm(total=total_length, desc="Detecting and Aligning faces: ") as pbar:
            for img_name in img_names:
                
                filepath = os.path.join(in_folder, identity, img_name)
                img = cv2.imread(filepath)

                boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)
        
                if landmarks is None:
                    skipped_imgs.append(img_name)
                    continue
        
                facial5points = landmarks[0]
                facial5points = np.array(landmarks[0], dtype=np.float32) 
                
                warped_face = norm_crop(img, landmark=facial5points, createEvalDB=True)
                # cv2.imwrite(os.path.join(out_folder, identity, img_name), cv2.cvtColor(warped_face, cv2.COLOR_BGR2RGB))
                cv2.imwrite(os.path.join(out_folder, identity, img_name), warped_face)
                pbar.update(1)

    print(skipped_imgs)
    print(f"Images with no Face: {len(skipped_imgs)}")

def move_failed_to_aligned(from_folder):
    to_folder = "data\\aligned faces"
    for filename in os.listdir(from_folder):
        shutil.copy(f"{from_folder}/{filename}", f"{to_folder}/{filename}")
        print(f"Copied {filename}")

def align_faces_gpu(in_folder,out_folder,gpu):
    # in_folder = "data\\Failed detected images"      # path to input images
    # out_folder = r"data\\aligned_for_failed"        # path to save aligned images
    # gpu = 0                                         # GPU ID (use -1 for CPU)

    align_images(in_folder, out_folder, gpu)

    move_failed_to_aligned(f"{out_folder}\\New folder")

