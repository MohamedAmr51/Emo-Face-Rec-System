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

# Emotions labels
emotions = ['happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'neutral']
people_name = {
        }

people_name_G8 = {
        'person_0':'Eyad Masoud',
        'person_1':'Khaled Mahmoud', 
        'person_3':'Habiba Gharabawy',
        'person_6':'Abdelrahman Azabawy',
        'person_7':'Zeyad El-Sheikh',
        'person_8':'Mohamed Fahmy',
        'person_14':'Hala Refaey',
        'person_17':'Helen Bakhoum',
        'person_43':'Mohamed Tawfeek'
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
fer_simple_results = []

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

def process_emotions_all_temp_folders(persons_temp_dir , persons_dir):
    """Process emotions for all person temp folders"""
    if not os.path.exists(persons_temp_dir) or not os.path.exists(persons_dir):
        return
    
    for person_folder in os.listdir(persons_dir):
        if not person_folder.startswith("person_"):
            continue
            
        person_path = os.path.join(persons_dir, person_folder)
        person_temp_path = os.path.join(persons_temp_dir,person_folder)

        if len(os.listdir(person_path)) < 4 :
            continue 

        temp_path = os.path.join(person_temp_path, "temp")

        if not os.path.exists(temp_path):
            continue
        
        # Get all face images from temp folder
        temp_faces = [f for f in os.listdir(temp_path) 
                     if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if not temp_faces:
            continue
        
        person_emotions = []
        person_confidences = []
        face_emotion_map = {}  # Track emotion for each face
        face_confidence_map = {} # Track Confidence for each face

        temp_faces_copy = temp_faces.copy()
        # Process each face for emotion
        for face_file in temp_faces_copy:
            face_path = os.path.join(temp_path, face_file)
            image = cv.imread(face_path)
            
            if image is not None:
                gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                pil_img = Image.fromarray(gray_image)
                emotion_data = detect_emotion(pil_img)
                
                detected_emotion = emotion_data['detected_emotion']

                if detected_emotion in ["fear" , "anger" , "disgust"]:
                    temp_faces.remove(face_file)
                    continue
                elif detected_emotion in ["surprise"]:
                    detected_emotion = "happy"
                confidence = emotion_data['confidence_score']
                
                person_emotions.append(detected_emotion)
                person_confidences.append(confidence)
                face_emotion_map[face_file] = detected_emotion
                face_confidence_map[face_file] = confidence

                person_name = people_name.get(person_folder, person_folder)
                fer_simple_results.append({
                'timestamp': extract_face_timestamp(face_file),
                'person_id' : person_name,
                'detected_emotion': detected_emotion,
                'confidence' : confidence,
                'happy_prob' : emotion_data['happy_prob'],
                'surprise_prob': emotion_data['surprise_prob'],
                'sad_prob': emotion_data['sad_prob'],
                'anger_prob': emotion_data['anger_prob'],
                'disgust_prob': emotion_data['disgust_prob'],
                'fear_prob': emotion_data['fear_prob'],
                'neutral_prob':emotion_data['neutral_prob']
                })
                
        if not person_emotions:
            continue
        
        # Move emotion faces to predictions folder
        predictions_person_folder_simple = f"simple_predictions/{person_folder}"
        os.makedirs(predictions_person_folder_simple, exist_ok=True)
        
        moved_count = 0
        for face_file in temp_faces: 
            src_path = os.path.join(temp_path, face_file)
            dst_path = os.path.join(predictions_person_folder_simple , str(face_confidence_map[face_file]) + "_" + face_emotion_map[face_file] + "_"  + face_file )
            try:
                shutil.move(src_path, dst_path)
                moved_count += 1
            except Exception as e:
                print(f"Error copying {face_file}: {e}")
        

        print(f"Processed {person_folder}: moved {moved_count} faces")

    if fer_simple_results:
        append_simple_csv_records(fer_simple_results)
        print(f"Processed batch of {len(fer_simple_results)} person records")
        
        # Clear the results list to free memory
        fer_simple_results.clear()
        print("fer_simple_results list cleared from memory")
      
    else:
        print("No faces accumulated to process.")

def extract_face_timestamp(face_file): # used to extract timestamp from image path
    """Extract and convert Unix timestamp from face filename"""
    # face_0_1753018669_3929.jpg -> extract 1753018669
    parts = face_file.split('_')
    if len(parts) >= 3:
        try:
            unix_timestamp = int(parts[2])
            return datetime.fromtimestamp(unix_timestamp).strftime('%Y-%m-%d %H:%M:%S')
        except (ValueError, OSError):
            return None
    return None

def append_simple_csv_records(fer_simple_results):
    if not fer_simple_results:
       print("No simple results to append")
       return
    try:
        simple_csv_filename = 'C:\\Users\\Admin.Amr\\OneDrive - Misr Italia properties\\G8 Fer\\simple_fer_results_90st.csv'
        file_exists = os.path.exists(simple_csv_filename)

        # Convert new results to DataFrame
        new_df = pd.DataFrame(fer_simple_results)

        # Save the data with proper quoting
        new_df.to_csv(simple_csv_filename , mode='a' , header=not file_exists, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"Appended to simple CSV file with {len(new_df)} records")

    except Exception as e:
        print(f"Error appending to Simple CSV: {e}")
    
    return

# process_emotions_all_temp_folders()