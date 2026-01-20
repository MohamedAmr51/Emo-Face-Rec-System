"""
CPU-based Face Alignment Utility.

This module provides functions to align faces using the SER-FIQ wrapper (likely wrapping MTCNN)
running on the CPU. It is used as a fallback or alternative to the GPU-based alignment.

Key Features:
- Iterates through a directory of raw face images.
- Aligns faces and saves them to a structured directory.
- Handles failures by moving problematic images to a separate folder.
"""
from utils.face_image_quality import SER_FIQ
import cv2
import os 
import numpy as np
from tqdm import tqdm

def align_faces_cpu(faces_dir):
    """
    Align faces in a directory using the CPU.
    
    Args:
        faces_dir (str): Directory containing raw face images.
        
    Workflow:
    1. Initializes SER_FIQ model (CPU mode).
    2. Creates output directories for aligned and failed images.
    3. Iterates through `faces_dir`.
    4. Aligns each image.
    5. Saves successful alignments to `data/aligned faces`.
    6. Moves failed alignments to `data/Failed detected images/New folder`.
    """
    # Create the SER-FIQ Model
    ser_fiq = SER_FIQ(gpu=None)
    
    os.makedirs("data\\Failed detected images\\New folder",exist_ok=True)
    os.makedirs("data\\aligned faces",exist_ok=True)

    with tqdm(total=len(os.listdir(faces_dir)), desc=" applying Mtcnn model : ") as pbar:
        for filename in os.listdir(faces_dir):
            if filename.endswith(".jpg"): 

                image_path = os.path.join(faces_dir,filename)   
                # Load the test image
                test_img = cv2.imread(image_path)
                
                # Align the image
                aligned_img = ser_fiq.apply_mtcnn(test_img)

                if aligned_img is None:
                    print(f"MTCNN alignment failed for {filename} .")
                    out_path = os.path.join("data\\Failed detected images\\New folder" , filename)
                    cv2.imwrite(out_path , test_img)
                    pbar.update(1)
                    continue
                
                aligned_img_save = np.transpose(aligned_img, (1, 2, 0))
                
                # Convert from RGB back to BGR for OpenCV saving
                aligned_img_save = cv2.cvtColor(aligned_img_save, cv2.COLOR_RGB2BGR)

                out_path = os.path.join("data\\aligned faces" , filename)
                cv2.imwrite(out_path , aligned_img_save)
                pbar.update(1)