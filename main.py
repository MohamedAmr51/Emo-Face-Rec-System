import os
import sys
import cv2 as cv
import io
import csv
import glob
import time
import shutil
import random
import argparse
import contextlib
import threading
import numpy as np
import pandas as pd
import cv2.data
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
from deepface import DeepFace
from collections import Counter
from stream_handler import ThreadedVideoCapture
from emo_rec import process_emotions_all_temp_folders


accumulated_faces = []  # Global list to accumulate faces across frames
# new_persons_created = False  # Track if new persons were created

# To set Functions start time
face_rec_sec = 5                    #recognize detected faces every n seconds
outlier_seconds = 120               #delete outlier people every n seconds
fer_seconds = 20                   #call FER model every n seconds
similarity_threshold=0.40           #select similarity threshold for comparison between persons
outlier_n_photos = 4                #select no. of photos to decide if outlier 

# To specify size of faces (width and height for the detected face)
min_w = 30
min_h = 30

# To set zoom intensity 
zoom_factor = 2

# 90st Waiting Area
WA_x = 250
WA_y = 50

# Management Area
MA_x = 150 
MA_y = 100

def create_main_folders():
    """For Creating folder first time"""
    global persons_dir                      # Set to global for assign_faces_to_person func.
    persons_dir = "Persons_Faces"           # Dir to save persons inside it
    if not os.path.exists(persons_dir):
        os.mkdir(persons_dir)

def delete_outliers():
    """Deleting the person folder if length of folder less than the required length"""
    for person_id in os.listdir(persons_dir):
        person_folder = os.path.join(persons_dir,person_id)
        try:
            if len(os.listdir(person_folder)) <=outlier_n_photos +1 :
                shutil.rmtree(person_folder)
                print(f"Deleted outlier {person_id} .")
        except:
            continue

def find_best_matching_person(face_img):

    # Use DeepFace.find
    results = DeepFace.find(
        img_path=face_img,              # input face image
        db_path=persons_dir,            # database directory
        model_name="Facenet512",        # model name for recognition
        distance_metric="cosine",           
        enforce_detection=False,        # don't raise error if face not detected
        silent=False,                    # Suppress logs
        refresh_database= True
        # detector_backend="retinaface"          # refresh database for new imported faces
    )
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
        print ("Matched Person:",matched_person,"Similarity:",similarity)
        
        if matched_person:
            # Assign to existing person
            person_folder = os.path.join(persons_dir, matched_person)
            temp_folder = os.path.join(person_folder, "temp")
        else:
            # Create new person
            global new_persons_created 
            new_persons_created = True  
            matched_person = f"person_{next_person_id}"
            existing_persons.add(next_person_id)  
            person_folder = os.path.join(persons_dir, matched_person)
            temp_folder = os.path.join(person_folder, "temp")
            
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
            face_filename = f"{person_folder}/face_{face_data['face_idx']}_{face_data['frame_timestamp']}.jpg"
            cv.imwrite(face_filename, face_image)
        
        # Save copy to temp folder (for emotion analysis)
        temp_face_filename = f"{temp_folder}/face_{face_data['face_idx']}_{face_data['frame_timestamp']}.jpg"
        cv.imwrite(temp_face_filename, face_image)

def zoom_crop(img_np):
    """
    Zoom into a manually defined region in the image.
    Input: img_np = OpenCV image (NumPy array, BGR)
    Returns: zoomed image as NumPy array (still in BGR for OpenCV)
    """
    # Convert BGR (OpenCV) to RGB (PIL expects RGB)
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image
    img = Image.fromarray(img_rgb)

    w, h = img.size
    # ==================================

    crop_w = w // zoom_factor
    crop_h = h // zoom_factor

    end_x = min(WA_x + crop_w, w)
    end_y = min(WA_y + crop_h, h)

    box = (WA_x, WA_y, end_x, end_y)
    cropped = img.crop(box)

    # Resize back to original size
    zoomed = cropped.resize((w, h), Image.LANCZOS)

    # Convert back to NumPy (RGB)
    zoomed_np = np.array(zoomed)

    # Convert RGB back to BGR for OpenCV
    zoomed_bgr = cv2.cvtColor(zoomed_np, cv2.COLOR_RGB2BGR)
    return zoomed_bgr


while True:
        current_time = time.perf_counter()
        # Process accumulated faces when counter reaches threshold or we have enough faces
        if  current_time - last_face_rec_time >= face_rec_sec or len(accumulated_faces) >= 100: 
            elapsed_time = current_time - last_face_rec_time
            print(f"face_rec loop started after {elapsed_time:.2f} seconds")

            if accumulated_faces:
                print(f"Processing {len(accumulated_faces)} accumulated faces...")

                # Assign faces to persons 
                assign_faces_to_persons(accumulated_faces)
                
                # Clear accumulated faces after assignment
                accumulated_faces.clear()

            last_face_rec_time = current_time
        
        # Delete Outliers every 1.5 minute
        if current_time - outliers_time >= outlier_seconds:
            print(f"Deleting Outliers after {current_time-outliers_time}")
            delete_outliers()
            print("Deleting Outliers Finished.")
            outliers_time = current_time 

        if current_time-last_emo_rec_time >= fer_seconds:
            try:
                # Process emotions for each person's temp folder
                process_emotions_all_temp_folders()
            except Exception as e:
                print(f"Error during emotion processing: {e}")
                continue

            last_emo_rec_time = current_time
                
    
    cap.stop()