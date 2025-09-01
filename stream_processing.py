import time
import cv2 as cv
import contextlib
from deepface import DeepFace
from stream_handler import ThreadedVideoCapture

frame_detect_milli_sec = 80        #receive frames every n milliseconds
expand_percent = 20                 #to increase the space around the detected face

def start_stream(stream):
        cap = ThreadedVideoCapture(stream)
        cap.start()
        frame_count = 0 

        while cv.waitKey(frame_detect_milli_sec) < 0:
            frame = cap.get_frame()
            if frame is None:
                print('No frame available!')
                continue

            frame_count += 1
            if frame_count % 5 == 0:  # Process every 5th frame
                continue

            # cv.imwrite("test.jpg",frame)

            with contextlib.redirect_stdout(io.StringIO()):
                results = DeepFace.extract_faces(
                    img_path=frame,
                    detector_backend="mtcnn",
                    enforce_detection=False , 
                    expand_percentage = expand_percent
                )
            tm.stop()
            timestamp = f"{int(time.time())}_{frame_count}"
            
            # Accumulate faces across frames for better clustering
            for idx in range(len(results)):
                x = results [idx] ["facial_area"]["x"]
                y = results [idx] ["facial_area"]["y"]
                w = results [idx] ["facial_area"]["w"]
                h = results [idx] ["facial_area"]["h"]
                if w > 0 and h > 0 and w > min_w and h > min_h and results [idx]["confidence"] == 1.00 :
                    face_crop = frame[y:y+h, x:x+w]
                    accumulated_faces.append({
                        'image': face_crop,
                        'frame_timestamp': timestamp,
                        'face_idx': idx,
                        'frame_count': frame_count
                    })
            
    return 
