import cv2 as cv
import threading
import time

class ThreadedVideoCapture:
    def __init__(self, src=0):
        self.capture = cv.VideoCapture(src)
        self.capture.set(cv.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer
        self.thread = None
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        
    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._update_frame)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()
        self.capture.release()
        
    def _update_frame(self):
        while self.running:
            ret, frame = self.capture.read()
            if ret:
                with self.lock:
                    self.frame = frame
            time.sleep(0.01)  # Small delay to prevent 100% CPU usage
            
    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
            
    def is_opened(self):
        return self.capture.isOpened()
