"""
GPU-accelerated Face Alignment Utility.

This module provides functions to align faces using the facenet_pytorch MTCNN detector
running on GPU for faster processing.

Key Features:
- Detects facial landmarks using MTCNN.
- Applies geometric normalization (warping) to align faces.
- Handles batch processing of person-specific folders.
- Moves processed images to the aligned faces directory.
"""
import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import shutil 

from utils.align_trans import norm_crop

from facenet_pytorch import MTCNN

def align_images(in_folder, out_folder, gpu):
    """
    Align all images in a folder structure using MTCNN.
    
    Args:
        in_folder (str): Input directory containing person folders with images.
        out_folder (str): Output directory to save aligned images.
        gpu (int): GPU device ID (e.g., 0 for cuda:0).
        
    Workflow:
    1. Initializes MTCNN on the specified GPU.
    2. Iterates through person folders.
    3. Detects facial landmarks for each image.
    4. Applies geometric normalization (warping).
    5. Saves aligned faces to the output folder.
    """  
    mtcnn = MTCNN(select_largest=True, post_process=False, device=gpu)
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
                cv2.imwrite(os.path.join(out_folder, identity, img_name), warped_face)
                pbar.update(1)

    print(skipped_imgs)
    print(f"Images with no Face: {len(skipped_imgs)}")

def move_failed_to_aligned(from_folder):
    """
    Move processed images from a temporary folder to the main aligned faces directory.
    
    Args:
        from_folder (str): Source folder containing processed images.
    """
    to_folder = "data\\aligned faces"
    for filename in os.listdir(from_folder):
        shutil.copy(f"{from_folder}/{filename}", f"{to_folder}/{filename}")
        print(f"Copied {filename}")

def align_faces_gpu(in_folder, out_folder, gpu):
    """
    Main entry point for GPU-based face alignment.
    
    Args:
        in_folder (str): Input directory with raw images.
        out_folder (str): Output directory for aligned images.
        gpu (int): GPU device ID.
        
    This function orchestrates the alignment process and moves the results.
    """                                       

    align_images(in_folder, out_folder, gpu)
    move_failed_to_aligned(f"{out_folder}\\New folder")

