import io
import os
import time
import cv2 as cv
import contextlib
from deepface import DeepFace
from stream_handler import ThreadedVideoCapture

frame_detect_milli_sec = 80         #receive frames every n milliseconds
expand_percent = 20                 #to increase the space around the detected face

# To specify size of faces (width and height for the bounding box of detected face)
min_w = 40
min_h = 40

# Folder path for quality model
quality_path = "Quality Assessment\\data\\Detected Faces\\New folder"

def save_image(face_data):
    """
    Save the image face into the quality folder for quality checking and alignment
    """
  
    os.makedirs(quality_path, exist_ok=True)
    face_filename = f"{quality_path}\\face_{face_data['face_idx']}_{face_data['frame_timestamp']}.jpg"
    cv.imwrite(face_filename, face_data['image'])

def stream_worker(stream , face_queue , worker_id , stop_event):
        print(f"Worker {worker_id} starting for stream: {stream}")

        cap = ThreadedVideoCapture(stream)
        cap.start()
        frame_count = 0 
        try:
            while not stop_event.is_set():
                frame = cap.get_frame()
                if frame is None:
                    print(f'Worker {worker_id}: No frame available!')
                    time.sleep(0.1)
                    continue

                frame_count += 1
                if frame_count % 5 == 0:  # Process every 5th frame
                    continue
                
                # cv.imwrite("test.jpg",frame)
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        results = DeepFace.extract_faces(
                            img_path=frame,
                            detector_backend="mtcnn",
                            enforce_detection=False , 
                            expand_percentage = expand_percent
                        )
                    timestamp = f"{int(time.time())}_{frame_count}_{worker_id}"
                    
                    # Accumulate faces across frames for better clustering
                    for idx in range(len(results)):
                        x = results [idx] ["facial_area"]["x"]
                        y = results [idx] ["facial_area"]["y"]
                        w = results [idx] ["facial_area"]["w"]
                        h = results [idx] ["facial_area"]["h"]

                        if w > 0 and h > 0 and w > min_w and h > min_h and results [idx]["confidence"] == 1.00 :
                            face_crop = frame[y:y+h, x:x+w]
                            face_data = {
                                'image': face_crop,
                                'frame_timestamp': timestamp,
                                'face_idx': idx,
                                'frame_count': frame_count,
                                'worker_id':worker_id
                            }

                            save_image(face_data)

                except Exception as e:
                    print(f"Worker {worker_id}: Error processing frame: {e}")
                    continue
                time.sleep(frame_detect_milli_sec/1000)

        except Exception as e:
            print(f"Worker {worker_id}: Fatal error: {e}")
        finally:
            cap.stop()
            print(f"Worker {worker_id} stopped")