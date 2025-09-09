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
accumulated_faces_copy = []     # List to store temp faces for recognition

# Directories to save persons
global persons_dir , persons_temp_dir       # Set to global for assign_faces_to_person func.
persons_dir = "Persons_Faces_G8"            # Dir to save persons inside it
persons_temp_dir = "Persons_Temp_G8"

# To set Functions start time
face_rec_sec = 5                    #recognize detected faces every n seconds
outlier_seconds = 60               #delete outlier people every n seconds
fer_seconds = 20                   #call FER model every n seconds
similarity_threshold=0.4554           #select similarity threshold for comparison between persons
outlier_n_photos = 4                #select no. of photos to decide if outlier 

#DeepFace Parameters
face_recognition_model="ArcFace"
metric="cosine"
face_detection_model= "mtcnn"
expand_percent=1
alignment=True
norm="base"

def create_main_folders():
    """For Creating folder first time"""

    if not os.path.exists(persons_dir):
        os.mkdir(persons_dir)

    if not os.path.exists(persons_temp_dir):
        os.mkdir(persons_temp_dir)

def delete_outliers():
    """Deleting the person folder if length of folder less than the required length"""
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
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
    # Use DeepFace.find() instead of manual comparison
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
    try:
        with open(os.devnull, 'w') as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                # Use DeepFace.find
                results = DeepFace.find(
                    img_path=face_img,                          # input face image
                    db_path=persons_dir,                        # database directory
                    model_name=face_recognition_model,          # model name for recognition
                    detector_backend=face_detection_model,      # model name for detection
                    distance_metric="cosine",           
                    enforce_detection=True,                     # raise error if face not detected
                    silent=True,                                # Suppress logs
                    refresh_database= True                      # refresh database for new imported faces
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
    Assign faces to existing persons or create new ones
    Replaces the cluster_faces function
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
        if len(os.listdir(person_folder)) < 290:
            face_filename = f"{person_folder}/quality_{face_data['face_quality']}_face_{face_data['face_idx']}_{face_data['frame_timestamp']}.jpg"
            cv.imwrite(face_filename, face_image)
        
        # Save copy to temp folder (for emotion analysis)
        temp_face_filename = f"{temp_folder}/face_{face_data['face_idx']}_{face_data['frame_timestamp']}.jpg"
        cv.imwrite(temp_face_filename, face_image)
        
        try:
            temp = deepface_find(face_image)
            if temp:
                print("DB updated sucessfully .")
        except Exception as e:
            print(f"DB not updated successfully , {e}")

def main():
    """Main processing loop with queue-based multi-stream processing"""
    
    # RTSP streams configuration
    streams = [
        "rtsp://admin:LV@0000@lv@10.10.10.134/onvif/profile2/media.smp",  # Management Floor Cam 1
        "rtsp://admin:LV@0000@lv@10.10.10.135/onvif/profile2/media.smp"   # Management Floor Cam 2
        # Add more streams as needed
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
                    assign_faces_to_persons(faces_batch)
                    
                last_face_rec_time = current_time
            
            # Delete outliers periodically
            if current_time - outliers_time >= outlier_seconds:
                print(f"\nDeleting outliers after {current_time - outliers_time:.1f} seconds")
                delete_outliers()
                print("Outlier deletion completed")
                outliers_time = current_time
            
            # Process emotions periodically
            if current_time - last_emo_rec_time >= fer_seconds:
                try:
                    print("\nProcessing emotions for all persons...")
                    process_emotions_all_temp_folders(persons_temp_dir , persons_dir)
                    print("Emotion processing completed")
                except Exception as e:
                    print(f"Error during emotion processing: {e}")
                
                last_emo_rec_time = current_time
            
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