import numpy as np 
import cv2 as cv
from PIL import Image
import pandas as pd
from datetime import datetime
import pickle
import torch
import shutil
import csv
import os
import torch.nn.functional as F
from collections import Counter
import torchvision.transforms as transforms
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

global predictions_dir , persons_dir
predictions_dir = "predictions"
simple_predictions_dir = "simple_predictions"
persons_dir = "Persons_Faces"

# Emotions labels
emotions = ['happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'neutral']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

filename = 'D:\\Mohamed\\Desktop\\deep_face\\finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
fer_results = [] 
fer_simple_results = []

#Variables for sending the email
From = "Tempforbluestacks70@gmail.com"
To = ["Mohamed.ElSayed@misritaliaproperties.com" , "mohamedamr485@gmail.com"]
Subject = "Emotion Alert Test"
Username ="Tempforbluestacks70@gmail.com"
Userpassword="xwcppwquwscnablc"
Server = "smtp.gmail.com"
Port = 587

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def detect_emotion(image):
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = loaded_model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
    scores = probabilities.cpu().numpy().flatten()
    rounded_scores = [round(score, 2) for score in scores]
    max_index = np.argmax(rounded_scores)
    max_emotion = emotions[max_index]
    return {
        'detected_emotion': max_emotion,
        'confidence_score': rounded_scores[max_index],
        'happy_prob': rounded_scores[0],
        'surprise_prob': rounded_scores[1], 
        'sad_prob': rounded_scores[2],
        'anger_prob': rounded_scores[3],
        'disgust_prob': rounded_scores[4],
        'fear_prob': rounded_scores[5],
        'neutral_prob': rounded_scores[6]
    }

def process_emotions_all_temp_folders():
    """Process emotions for all person temp folders"""
    if not os.path.exists(persons_dir):
        return
    
    for person_folder in os.listdir(persons_dir):
        if not person_folder.startswith("person_"):
            continue
            
        person_path = os.path.join(persons_dir, person_folder)
        if len(os.listdir(person_path)) < 5 :
            continue 

        temp_path = os.path.join(person_path, "temp")

        if not os.path.exists(temp_path):
            continue
        if len(os.listdir(temp_path)) < 3 :
            shutil.rmtree(temp_path)
            print(f"Images wasn't enough for FER , Deleted temp images for {person_folder}.")
            continue
        
        # Get all face images from temp folder
        temp_faces = [f for f in os.listdir(temp_path) 
                     if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if not temp_faces:
            continue
        
        person_emotions = []
        person_confidences = []
        face_emotion_map = {}  # Track emotion for each face
        
        # Process each face for emotion
        for face_file in temp_faces:
            face_path = os.path.join(temp_path, face_file)
            image = cv.imread(face_path)
            
            if image is not None:
                gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                pil_img = Image.fromarray(gray_image)
                emotion_data = detect_emotion(pil_img)
                
                detected_emotion = emotion_data['detected_emotion']
                if detected_emotion in ["surprise","fear"]:
                    detected_emotion = "happy"
                elif detected_emotion in ["anger","disgust"]:
                    detected_emotion = "sad"
                confidence = emotion_data['confidence_score']
                
                person_emotions.append(detected_emotion)
                person_confidences.append(confidence)
                face_emotion_map[face_file] = detected_emotion
        
        if not person_emotions:
            continue
        
        # Calculate dominant emotion
        emotion_counts = Counter(person_emotions)
        dominant_emotion = emotion_counts.most_common(1)[0][0]
        dominant_count = emotion_counts[dominant_emotion]
        dominant_frequency = (dominant_count / len(person_emotions)) * 100
        
        # Average confidence for dominant emotion only
        dominant_confidences = [conf for emotion, conf in zip(person_emotions, person_confidences) 
                              if emotion == dominant_emotion]
        avg_confidence = sum(dominant_confidences) / len(dominant_confidences) if dominant_confidences else 0.0
        
        # Move only dominant emotion faces to predictions folder
        predictions_person_folder = f"predictions/{dominant_emotion}/{person_folder}"
        os.makedirs(predictions_person_folder, exist_ok=True)
        
        moved_count = 0
        for face_file in temp_faces:
            if face_emotion_map[face_file] == dominant_emotion:
                src_path = os.path.join(temp_path, face_file)
                dst_path = os.path.join(predictions_person_folder, face_file )
                try:
                    shutil.move(src_path, dst_path)
                    moved_count += 1
                except Exception as e:
                    print(f"Error moving {face_file}: {e}")
        
        # Clean up temp folder (remove any remaining files)
        try:
            for remaining_file in os.listdir(temp_path):
                os.remove(os.path.join(temp_path, remaining_file))
            print(f"Cleared temp folder for {person_folder}")
        except Exception as e:
            print(f"Error clearing temp folder for {person_folder}: {e}")
        
        # Record results
        fer_results.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'person_id': person_folder,
            'dominant_emotion': dominant_emotion,
            'dominant_frequency_percent': round(dominant_frequency, 2),
            'average_confidence': round(avg_confidence, 3),
            'total_faces': len(person_emotions),
            'emotion_distribution': dict(emotion_counts),
            'faces_moved_to_predictions': moved_count
        })
        
        print(f"Processed {person_folder}: {dominant_emotion} ({dominant_frequency:.1f}%), moved {moved_count} faces")

    if fer_results:
        update_person_csv_records(fer_results)
        print(f"Processed batch of {len(fer_results)} person records")
        
        # Clear the results list to free memory
        fer_results.clear()
        print("fer_results list cleared from memory")
        
    else:
        print("No faces accumulated to process.")

def update_person_csv_records(fer_results):
    """
    Update existing person records in CSV and maintain history
    """
    csv_filename = 'fer_results.csv'
    history_filename = 'fer_results_history.csv'
    
    # Read existing data if file exists
    existing_df = pd.DataFrame()
    if os.path.exists(csv_filename):
        try:
            existing_df = pd.read_csv(csv_filename)
        except Exception as e:
            print(f"Error reading existing CSV: {e}")
            existing_df = pd.DataFrame()
    
    # Convert new results to DataFrame
    new_df = pd.DataFrame(fer_results)
    
    if existing_df.empty:
        # First time - just save the new data with proper quoting
        new_df.to_csv(csv_filename, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"Created new CSV file with {len(new_df)} records")
        return
    
    # Keep track of updated records for history
    history_records = []
    
    # Process each new result
    for _, new_record in new_df.iterrows():
        person_id = new_record['person_id']
        
        # Check if person already exists
        person_mask = existing_df['person_id'] == person_id
        
        if person_mask.any():
            # Person exists - check if update is needed
            existing_record = existing_df[person_mask].iloc[0]
            
            # Compare key fields to see if there's a significant change
            emotion_changed = existing_record['dominant_emotion'] != new_record['dominant_emotion']
            frequency_changed = abs(existing_record['dominant_frequency_percent'] - new_record['dominant_frequency_percent']) > 5.0
            confidence_changed = abs(existing_record['average_confidence'] - new_record['average_confidence']) > 0.1
            faces_increased = new_record['total_faces'] > existing_record['total_faces']
            
            if emotion_changed or frequency_changed or confidence_changed or faces_increased:
                # Save old record to history before updating
                history_record = existing_record.copy()
                update_reasons = []
                
                if emotion_changed:
                    update_reasons.append(f"emotion_change_{existing_record['dominant_emotion']}_to_{new_record['dominant_emotion']}")
                if frequency_changed:
                    update_reasons.append(f"frequency_change_{existing_record['dominant_frequency_percent']}_to_{new_record['dominant_frequency_percent']}")
                if confidence_changed:
                    update_reasons.append(f"confidence_change_{existing_record['average_confidence']}_to_{new_record['average_confidence']}")
                if faces_increased:
                    update_reasons.append(f"faces_increased_{existing_record['total_faces']}_to_{new_record['total_faces']}")
                
                # FIX: Use semicolon instead of comma to avoid CSV parsing issues
                history_record['update_reason'] = '; '.join(update_reasons)
                history_record['archived_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                history_records.append(history_record)
                
                # Update the existing record with cumulative data
                updated_record = new_record.copy()
                
                # Merge emotion distributions
                try:
                    old_distribution = eval(existing_record['emotion_distribution']) if isinstance(existing_record['emotion_distribution'], str) else existing_record['emotion_distribution']
                    new_distribution = eval(new_record['emotion_distribution']) if isinstance(new_record['emotion_distribution'], str) else new_record['emotion_distribution']
                    
                    # Combine emotion counts
                    combined_distribution = {}
                    all_emotions = set(old_distribution.keys()) | set(new_distribution.keys())
                    
                    for emotion in all_emotions:
                        combined_distribution[emotion] = old_distribution.get(emotion, 0) + new_distribution.get(emotion, 0)
                    
                    updated_record['emotion_distribution'] = str(combined_distribution)
                    
                    # Recalculate dominant emotion and frequency from combined data
                    total_combined_faces = sum(combined_distribution.values())
                    dominant_emotion = max(combined_distribution, key=combined_distribution.get)
                    dominant_count = combined_distribution[dominant_emotion]
                    
                    updated_record['dominant_emotion'] = dominant_emotion
                    updated_record['dominant_frequency_percent'] = round((dominant_count / total_combined_faces) * 100, 2)
                    updated_record['total_faces'] = total_combined_faces
                    
                except Exception as e:
                    print(f"Error merging emotion distributions for {person_id}: {e}")
                    # Keep the new record as is if merging fails
                
                # Add cumulative tracking fields
                updated_record['total_updates'] = existing_record.get('total_updates', 0) + 1
                updated_record['first_detected'] = existing_record.get('first_detected', existing_record['timestamp'])
                updated_record['last_updated'] = new_record['timestamp']

                """" This part is responsible for checking if the person has been angry or sad for a period of time
                then gets the latest images and send it to the user """
                # # Calculate how long the emotion has persisted
                # try:
                #     fd_dt = datetime.strptime(str(updated_record['first_detected']), '%Y-%m-%d %H:%M:%S')
                #     lu_dt = datetime.strptime(str(updated_record['last_updated']), '%Y-%m-%d %H:%M:%S')
                #     duration_minutes = round((lu_dt - fd_dt).total_seconds() / 60, 2)
                #     updated_record['emotion_duration_minutes'] = duration_minutes
                # except Exception as e:
                #     print(f"Error calculating duration for {person_id}: {e}")
                #     duration_minutes = 0.0
                # # Optional: define where to find or collect image paths
                # # You can replace this with your own logic (e.g., glob search)
                # person_images = []  # <-- replace with actual list of image paths for this person

                # latest_images_path = os.path.join(predictions_dir,updated_record['dominant_emotion'],person_id)
                # person_images = glob.glob(os.path.join(latest_images_path, "*.jpg"))[-3:]  # last 3 images
                # if updated_record['dominant_emotion'] in ['anger', 'sad'] and duration_minutes >= 3:
                #     try:
                #         SendMail(person_images, updated_record['dominant_emotion'], duration_minutes)
                #     except Exception as e:
                #         print(f"Error sending email for {person_id}: {e}")

                # FIX: Update the existing dataframe properly
                for col in updated_record.index:
                    if col in existing_df.columns:
                        existing_df.loc[person_mask, col] = updated_record[col]
                
                print(f"Updated person {person_id}: {updated_record['dominant_emotion']} ({updated_record['dominant_frequency_percent']}%)")
            else:
                print(f"No significant changes for person {person_id} - skipping update")
        else:
            # New person - add to existing dataframe
            new_person_record = new_record.copy()
            new_person_record['total_updates'] = 0
            new_person_record['first_detected'] = new_record['timestamp']
            new_person_record['last_updated'] = new_record['timestamp']
            
            existing_df = pd.concat([existing_df, new_person_record.to_frame().T], ignore_index=True)
            print(f"Added new person {person_id}: {new_record['dominant_emotion']}")
    
    # FIX: Save updated main CSV with proper quoting
    existing_df.to_csv(csv_filename, index=False, quoting=csv.QUOTE_NONNUMERIC)
    
    # FIX: Save history records with proper quoting
    if history_records:
        history_df = pd.DataFrame(history_records)
        
        # Append to history file with proper quoting
        file_exists = os.path.exists(history_filename)
        history_df.to_csv(history_filename, mode='a', header=not file_exists, index=False, quoting=csv.QUOTE_NONNUMERIC)
        
        print(f"Archived {len(history_records)} old records to history")
    
    print(f"Updated CSV with current data: {len(existing_df)} total persons")           

def SendMail(ImgFileNameList,emotion_type,period_time):
    
    Text = f"The following person has been {emotion_type} for the past {period_time} minutes."
    msg = MIMEMultipart()
    msg['Subject'] = Subject
    msg['From'] = From
    msg['To'] = ", ".join(To)
    mime_text = MIMEText(Text)
    msg.attach(mime_text)

    for ImgFileName in ImgFileNameList:
        with open(ImgFileName, 'rb') as f:
            img_data = f.read()
        image = MIMEImage(img_data, name=os.path.basename(ImgFileName))
        msg.attach(image)

    s = smtplib.SMTP(Server,Port)
    s.starttls()
    try:
        s.login(Username, Userpassword)
        s.sendmail(From, To, msg.as_string())
        print("Email Was Sent Successfully !")
        s.quit()
    except smtplib.SMTPAuthenticationError:
        print("UserName or Password are incorrect.")
