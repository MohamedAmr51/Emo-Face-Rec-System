"""
Continuous Face Quality Assessment Engine.

This module implements a real-time monitoring system that filters face images based on quality.
It uses the CR-FIQA (Face Image Quality Assessment) model to assign a score to each face.

Workflow:
1.  **Monitor**: Watches `data/Detected Faces/New folder` for new images.
2.  **Align**: Uses MTCNN (GPU/CPU) to align faces.
3.  **Prepare**: Calls `fixed_extract.py` to convert and list images.
4.  **Score**: Runs CR-FIQA inference to get a quality score (0-100+).
5.  **Filter**: Moves high-quality images (score >= 1.98) to `filtered_aligned_images_quality`.
6.  **Cleanup**: Removes processed and low-quality images to maintain a clean state.
"""
import argparse
import os
import sys
import time
import shutil 
import subprocess

from evaluation.QualityModel import QualityModel
from utils.serfiq_example import align_faces_cpu 
from utils.MTCNN_alignment_fast import align_faces_gpu

# Quality model initialization
face_model = QualityModel("./CR-FIQAL", "181952", 0)

# Paths for moving quality images
txt_file = "data\\quality_data\\custom_dataset\\custom_quality_scores.txt"   # Your text file
source_folder = r"data\\aligned faces"                                       # Folder where the images actually are
destination_folder = r"data\\filtered_aligned_images_quality"                # Folder where renamed copies will go

# Paths for image preprocessing
faces_dir = "data\\Detected Faces\\New folder"
in_folder = "data\\Failed detected images"      # path to input images
out_folder = r"data\\aligned_for_failed"        # path to save aligned images

wait_time = 10     # n seconds to check the raw faces folder
gpu = 0
processed_files = set()

def delete_leftovers(custom_data_dir):
    """
    Clean up temporary files and folders after a processing batch.
    
    Args:
        custom_data_dir (str): Path to the custom dataset directory to remove.
        
    Removes:
    - Failed detection images.
    - Intermediate alignment folders.
    - The temporary custom dataset folder used for scoring.
    - The source aligned faces folder.
    """
    faces_dir_files = os.listdir(faces_dir)
    for filename in os.listdir(in_folder):
        if filename in faces_dir_files:
            path_to_delete = os.path.join(faces_dir, filename)
            os.remove(path_to_delete)

    # Clear folders after quality detection
    try:
        shutil.rmtree(in_folder)
    except Exception as e:
        print(f"Error clearing {in_folder}: {e}")

    try:
        shutil.rmtree(out_folder)
    except Exception as e:
        print(f"Error clearing {out_folder}: {e}")

        # Delete custom data folder
    try:
        shutil.rmtree(custom_data_dir)
    except Exception as e:
        print(f"\nError clearing {custom_data_dir}: {e}")
    
    try:
        shutil.rmtree(source_folder)
    except Exception as e:
        print(f"Error clearing {source_folder}: {e}")

    print("\nCleared leftovers.")
    
def has_new_images():
    """
    Check if there are new images in the monitored directory.
    
    Returns:
        bool: True if new images are found, False otherwise.
    """
    global processed_files
    
    if not os.path.exists(faces_dir):
        return False
    
    # Get all image files in the directory
    current_files = set()
    for file in os.listdir(faces_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            current_files.add(file)
    
    new_files = current_files - processed_files
    
    if new_files:
        processed_files.update(new_files)
        print(f"Found {len(new_files)} new images")
        return True
    return False

def move_quality_images(path_list_file):
    """
    Filter and move images based on their quality score.
    
    Reads the generated score file, checks if the score meets the threshold (>= 1.98),
    and if so, renames and moves the file to the destination folder.
    Finally, calls `delete_leftovers` to clean up.
    
    Args:
        path_list_file (str): Path to the directory containing the score file.
    """
    # Make sure destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    # Read text file line by line
    with open(txt_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Split into filepath and score
            filepath, score = line.rsplit(" ", 1)
            score = float(score)  # Convert to float for formatting
            filename = os.path.basename(filepath)  # Extract filename only
            
            # Source file path (from source_folder)
            source_path = os.path.join(source_folder, filename)

            if score >= 1.98 :
                # New filename with score
                new_filename = f"quality_{score:.4f}_{filename}"  # Keep 4 decimals
                new_path = os.path.join(destination_folder, new_filename)

                source_path = os.path.join(source_folder, filename)
                
                try:
                    # Copy file if exists
                    if os.path.exists(source_path):
                        shutil.copy2(source_path, new_path)

                except Exception as e:
                    print(f"⚠️ File not found: {source_path} , Error: {e}")

            delete_path = os.path.join(faces_dir , filename)

            if os.path.exists(delete_path):
                os.remove(delete_path)
    
    delete_leftovers(path_list_file)
    
def parse_arguments(argv):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', type=str, default='/data/quality_data',
                        help='Root dir for evaluation dataset')
    parser.add_argument('--pairs', type=str, default='pairs.txt',
                        help='lfw pairs.')
    parser.add_argument('--datasets', type=str, default='custom_dataset',
                        help='list of evaluation datasets (,)  e.g.  custom_dataset, XQLFW, lfw,calfw,agedb_30,cfp_fp,cplfw,IJBC.')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU id.')
    # Fixed model path to match your structure
    parser.add_argument('--model_path', type=str, default="./CR-FIQAL",
                        help='path to pretrained evaluation.')
    parser.add_argument('--model_id', type=str, default="181952",
                        help='digit number in backbone file name')
    parser.add_argument('--backbone', type=str, default="auto",
                        help=' iresnet100, iresnet50, or auto to detect from weights ')
    parser.add_argument('--score_file_name', type=str, default="custom_quality_scores.txt",
                        help='score file name, the file will be store in the same data dir')
    parser.add_argument('--color_channel', type=str, default="RGB",  # Changed to RGB
                        help='input image color channel, two option RGB or BGR')

    return parser.parse_args(argv)

def read_image_list(image_list_file, image_dir=''):
    """Read the list of images to process from a text file."""
    image_lists = []
    absolute_list = []
    
    print(f"Reading image list from: {image_list_file}")
    
    with open(image_list_file) as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            absolute_list.append(line)
            # Create full path to image
            full_path = os.path.join(image_dir, line)
            image_lists.append(full_path)
    
    print(f"Found {len(image_lists)} images to process")
    return image_lists, absolute_list

def main(param):
    """
    Main execution loop.
    
    Continuously monitors for new images, runs alignment, extraction, and quality scoring.
    """
    d_data_dir = "data"
    datasets = param.datasets.split(',')
    print("Starting continuous monitoring...")
    
    while True:  # Infinite monitoring loop
        
        # Step 1: Check if there are new raw images to process
        if has_new_images():  # Check your input folders
            print("New raw images detected, starting preprocessing...")
            
            # Step 2: Face alignment (your essential steps)
            align_faces_cpu(faces_dir)
            align_faces_gpu(in_folder, out_folder, gpu)
            
            # Step 3: Extract and prepare data structure  
            subprocess.run(["python", "fixed_extract.py"])
            
            # Step 4: Quality assessment on the newly prepared data
            for dataset in datasets:
                dataset_dir = os.path.join(param.data_dir, dataset)
                image_list_file = os.path.join(dataset_dir, 'image_path_list.txt')
                
                if os.path.exists(image_list_file):
                    print(f"Running quality assessment on {dataset}...")
                    

                    image_list, absolute_list = read_image_list(image_list_file, d_data_dir)
                    embedding, quality = face_model.get_batch_feature(image_list, batch_size=16, color=param.color_channel)
                    
                    output_dir = os.path.join(param.data_dir, dataset)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    # Save results
                    quality_file_path = os.path.join(dataset_dir, param.score_file_name)
                    with open(quality_file_path, "w") as quality_score:
                        for i in range(len(quality)):
                            quality_score.write(f"{absolute_list[i]} {quality[i][0]}\n")
                    
                    print(f"Quality assessment complete! Results saved to {quality_file_path}")

                    move_quality_images(dataset_dir)
        else:
            print("No new raw images found, waiting...")
            
        time.sleep(wait_time)  # Wait for n seconds before next check

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))