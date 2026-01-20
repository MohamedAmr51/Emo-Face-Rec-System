import os
import sys
import cv2 as cv
import csv
import glob
import time
import shutil
import random
import numpy as np
import pandas as pd
from natsort import natsorted
import cv2.data
import matplotlib.pyplot as plt

from PIL import Image
from datetime import datetime
from deepface import DeepFace
from collections import Counter
from emo_rec import process_emotions_all_temp_folders
from contextlib import redirect_stdout, redirect_stderr
from multi_stream import MultiStreamProcessor

accumulated_faces = []          # Global list to accumulate faces across frames

# Directories to save persons
global persons_dir , persons_temp_dir       # Set to global for assign_faces_to_person func.
persons_dir = "Persons_Faces"               # Dir to save the person's recognition set
persons_temp_dir = "Persons_Temp"           # Dir to save the person's temp set for emotional detection

# ==========================================
# 1. Modifiable Parameters & Configuration
# ==========================================
# These parameters control the system's sensitivity, timing, and model choices.

# To set Functions start time
face_rec_sec = 5                                    # Interval (seconds) to run face recognition on accumulated faces
fer_seconds = 30                                    # Interval (seconds) to run Emotion Recognition on temp folders
similarity_threshold=0.4554                         # Cosine similarity threshold for ArcFace. Lower = stricter matching.
outlier_n_photos = 4                                # Min photos to keep a person folder. Variant based on detection/save frequency.
outlier_seconds = 120                               # Interval (seconds) to prune noise persons. Variant based on detection/save frequency.
max_person_photos = 290                             # Max photos to keep a person folder. Variant based on detection/save frequency.

# DeepFace Model Parameters
face_recognition_model="ArcFace"                    # Model used for generating face embeddings
metric="cosine"                                     # Distance metric for comparing embeddings
face_detection_model= "mtcnn"                       # Detector used by DeepFace (though a custom MTCNN is used in workers)
expand_percent=0                                    # Percentage to expand the bounding box
alignment=True                                      # Whether to align faces (eyes horizontally) before recognition
norm="base"                                         # Normalization technique for input images


def create_main_folders():
    """For Creating folders first time"""

    if not os.path.exists(persons_dir):
        os.mkdir(persons_dir)

    if not os.path.exists(persons_temp_dir):
        os.mkdir(persons_temp_dir)

def delete_outliers():
    """
    2. Outlier Deletion Function
    
    Periodically cleans up the 'Persons' directory to remove noise.
    
    Logic:
    - Iterates through all persons folders.
    - If a folder has fewer images than `outlier_n_photos` (e.g., 4), it is considered
      a false positive or a transient detection (someone walking by too fast).
    - Deletes both the main person folder and their corresponding temp folder.
    
    This prevents the system from accumulating thousands of "junk" identities over time.
    """
    for person_id in os.listdir(persons_dir):
        person_folder = os.path.join(persons_dir,person_id)
        try:
            if len(os.listdir(person_folder)) <=outlier_n_photos :
                person_temp_folder = os.path.join(persons_temp_dir,person_id)
                shutil.rmtree(person_folder)
                shutil.rmtree(person_temp_folder)
                print(f"Deleted outlier {person_id} .")
        except:
            continue

def deepface_find(face_img):
    """
    ===========================================================================
    NOTE: This function is primarily used to REFRESH the DeepFace database.
    By calling DeepFace.find with refresh_database=True, we ensure that the 
    internal embeddings representations are updated with any new person 
    folders or images added since the last run.
    ===========================================================================
    """
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            result = DeepFace.find(
                img_path=face_img,                              # input face image
                db_path=persons_dir,                            # database directory
                model_name=face_recognition_model,              # model name for recognition
                detector_backend=face_detection_model,          # model name for detection
                distance_metric="cosine",           
                enforce_detection=True,                         # raise error if face not detected
                silent=True,                                    # Suppress logs
                refresh_database= True                          # refresh database for new imported faces
        ) 
    return result

def find_best_matching_person(face_img):
    """
    Core recognition logic using DeepFace.
    
    Args:
        face_img: The face image to recognize.
        
    Returns:
        tuple: (person_id, similarity_score) or (None, 0) if no match found.
        
    Logic:
    1. Calls `DeepFace.find` to compare the input image against the entire database (`persons_dir`).
    2. Uses 'ArcFace' model and 'cosine' distance.
    3. If a match is found, it checks if the similarity score is above our threshold.
    4. Returns the ID of the best match.
    """
    try:
        results = DeepFace.find(
            img_path=face_img,                          # input face image
            db_path=persons_dir,                        # database directory
            model_name=face_recognition_model,          # model name for recognition
            detector_backend=face_detection_model,      # model name for detection
            distance_metric="cosine",           
            enforce_detection=True,                     # raise error if face not detected
            silent=True,                                # Suppress logs
            refresh_database= True,                     # refresh database for new imported faces
            expand_percentage=expand_percent
        )
    except Exception as e:
        print(f"No Face Detected. {e}")
        return "NO_FACE_DETECTED", -1
    # Handle the results - DeepFace returns a list of DataFrames
    if not results or len(results) == 0:
        return None, 0
    
    # Get the first DataFrame (assumes single face in input image)
    df = results[0]
    
    # Check if any matches were found
    if df.empty :
        return None, 0
    
    df['similarity'] = 1 - df['distance']
    
    # just check if the first result meets our threshold
    if df.iloc[0]['similarity'] < similarity_threshold:
        return None, 0
    
    # Extract person ID from the first (best) match
    best_match_path = df.iloc[0]['identity']
    person_id = best_match_path.split("\\")[1]
    best_similarity = df.iloc[0]['similarity']
    
    return person_id , best_similarity

def assign_faces_to_persons(accumulated_faces):
    """
    4. Assign Faces to Persons (Clustering Logic)
    
    This function processes a batch of faces and decides whether they belong to an
    existing person or if a new identity should be created.
    
    Args:
        accumulated_faces: List of face data dictionaries (image, metadata).
        
    Logic:
    1. Iterates through each face in the batch.
    2. Calls `find_best_matching_person` to see if it matches anyone in the DB.
    3. If Match Found:
       - Adds the image to that person's folder.
    4. If No Match:
       - Creates a NEW person ID (e.g., person_101).
       - Creates new folders for this person.
    5. Also saves a copy to a 'temp' folder for Emotion Analysis.
    6. Updates the DeepFace database incrementally.
    """
    # Get next available person ID
    existing_persons = set()
    if os.path.exists(persons_dir):
        for folder in os.listdir(persons_dir):
            if folder.startswith("person_") and os.path.isdir(os.path.join(persons_dir, folder)):
                try:
                    person_num = int(folder.split("_")[1])
                    existing_persons.add(person_num)
                except:
                    continue
    
    next_person_id = 0
    while next_person_id in existing_persons:
        next_person_id += 1
    
    for face_data in accumulated_faces:
        face_image = face_data['image']
    
        # Find best matching person
        matched_person, similarity = find_best_matching_person(face_image)
        if matched_person == "NO_FACE_DETECTED":
                print("Skipping image - no face detected")
                continue  # Skip to next iteration
        print ("Matched Person:",matched_person,"Similarity:",similarity)
        
        if matched_person:
            # Assign to existing person
            person_folder = os.path.join(persons_dir, matched_person)

            person_temp_folder = os.path.join(persons_temp_dir,matched_person)
            temp_folder = os.path.join(person_temp_folder, "temp")
        else:
            # Create new person 
            matched_person = f"person_{next_person_id}"
            existing_persons.add(next_person_id)  
            person_folder = os.path.join(persons_dir, matched_person)
            person_temp_folder = os.path.join(persons_temp_dir,matched_person)

            temp_folder = os.path.join(person_temp_folder, "temp")
            
            # Create folders
            os.makedirs(person_folder, exist_ok=True)
            os.makedirs(temp_folder, exist_ok=True)

            next_person_id += 1
            while next_person_id in existing_persons:
                next_person_id += 1
            
            print(f"Created new person: {matched_person}")
        
        # Ensure temp folder exists
        os.makedirs(temp_folder, exist_ok=True)
        
        # Save face image to person folder 
        if len(os.listdir(person_folder)) < max_person_photos :
            face_filename = f"{person_folder}/quality_{face_data['face_quality']}_face_{face_data['face_idx']}_{face_data['frame_timestamp']}_{face_data['frame_count']}_{face_data['worker_id']}.jpg"
            cv.imwrite(face_filename, face_image)
        
        # Save copy to temp folder (for emotion analysis)
        temp_face_filename = f"{temp_folder}/face_{face_data['face_idx']}_{face_data['frame_timestamp']}_{face_data['frame_count']}_{face_data['worker_id']}.jpg"
        cv.imwrite(temp_face_filename, face_image)
        
        try:
            temp = deepface_find(face_image)
            if temp:
                print("DB updated sucessfully .")
        except Exception as e:
            print(f"DB not updated successfully , {e}")

def main():
    """
    5. Main Processing Loop
    
    The central control loop of the application.
    
    Workflow:
    1. Configures RTSP streams.
    2. Starts the `MultiStreamProcessor` (background workers).
    3. Enters an infinite loop:
       - Checks the queue size.
       - If enough time passed (5s) OR queue is full:
         -> Pulls a batch of faces.
         -> Runs `assign_faces_to_persons` (Recognition).
       - If 120s passed:
         -> Runs `delete_outliers` (Maintenance).
       - If 30s passed:
         -> Runs `process_emotions_all_temp_folders` (Analytics).
    """
    
    # RTSP streams configuration
    streams = [
        # Add your RTSP streams here
    ]
    
    # Initialize folders
    create_main_folders()
    
    # Initialize multi-stream processor
    processor = MultiStreamProcessor(streams, max_queue_size=700)
    
    try:
        # Start all stream workers
        processor.start()
        
        # Initialize timing
        start_time = time.perf_counter()
        last_face_rec_time = start_time
        last_emo_rec_time = start_time
        outliers_time = start_time
        
        print("Starting main processing loop...")
        
        while True:
            current_time = time.perf_counter()
            queue_size = processor.get_queue_size()
            
            # Process faces when time threshold reached or queue getting full
            if (current_time - last_face_rec_time >= face_rec_sec or queue_size >= 100):
                elapsed_time = current_time - last_face_rec_time
                print(f"\nFace recognition cycle started after {elapsed_time:.2f} seconds")
                print(f"Queue size: {queue_size}")
                
                # Get batch of faces from queue
                faces_batch = processor.get_faces_batch(max_batch_size=50, timeout=2.0)
                
                if faces_batch:
                    print(f"Processing batch of {len(faces_batch)} faces...")
                    faces_batch = natsorted(faces_batch , key=lambda x: x['face_quality'], reverse=True)

                    assign_faces_to_persons(faces_batch)
                    
                last_face_rec_time = current_time
            
            # Delete outliers periodically
            if current_time - outliers_time >= outlier_seconds:
                print(f"\nDeleting outliers after {current_time - outliers_time:.1f} seconds")
                delete_outliers()
                print("Outlier deletion completed")
                outliers_time = current_time # Reset timer
            
            # Process emotions periodically
            if current_time - last_emo_rec_time >= fer_seconds:
                try:
                    print("\nProcessing emotions for all persons...")
                    process_emotions_all_temp_folders(persons_temp_dir , persons_dir)
                    print("Emotion processing completed")
                except Exception as e:
                    print(f"Error during emotion processing: {e}")
                
                last_emo_rec_time = current_time # Reset timer
            
            # Short sleep to prevent busy waiting
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nReceived interrupt signal...")
    except Exception as e:
        print(f"\nUnexpected error in main loop: {e}")
    finally:
        print("Stopping multi-stream processor...")
        processor.stop()
        print("System shutdown complete")

if __name__ == "__main__":
    main()