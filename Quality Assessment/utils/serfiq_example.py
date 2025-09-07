# Author: Jan Niklas Kolf, 2020
from utils.face_image_quality import SER_FIQ
import cv2
import os 
import numpy as np
from tqdm import tqdm

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

#Variables for sending the email
From = "Tempforbluestacks70@gmail.com"
To = ["Mohamed.ElSayed@misritaliaproperties.com" , "mohamedamr485@gmail.com"]
Username ="Tempforbluestacks70@gmail.com"
Userpassword="xwcppwquwscnablc"
Server = "smtp.gmail.com"
Port = 587

def SendMail(text , Subject):
    
    Text = f"{text}"
    msg = MIMEMultipart()
    msg['Subject'] = Subject
    msg['From'] = From
    msg['To'] = ", ".join(To)
    mime_text = MIMEText(Text)
    msg.attach(mime_text)

    s = smtplib.SMTP(Server,Port)
    s.starttls()
    try:
        s.login(Username, Userpassword)
        s.sendmail(From, To, msg.as_string())
        print("Email Was Sent Successfully !")
        s.quit()
    except smtplib.SMTPAuthenticationError:
        print("UserName or Password are incorrect.")

def align_faces_cpu(faces_dir):
    # Create the SER-FIQ Model
    ser_fiq = SER_FIQ(gpu=None)
    
    # faces_dir = "data\\original images\\New folder"
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