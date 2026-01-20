"""
Threaded Video Capture Module.

This module implements a non-blocking video capture class using threading.
Standard `cv2.VideoCapture.read()` is blocking, meaning if the camera is slow
or the network (RTSP) lags, frames may be dropped leading to inconsistent quality.

This class solves that by:
1.  Running the frame reading loop in a separate background thread (Daemon).
2.  Continuously grabbing the *latest* frame and discarding older ones.
3.  Allowing the main application to `get_frame()` instantly without waiting.
"""
import cv2 as cv
import threading
import time

class ThreadedVideoCapture:
    """
    A wrapper around OpenCV's VideoCapture that runs in a separate thread.
    """
    def __init__(self, src=0):
        """
        Initialize the video capture object.
        """
        self.capture = cv.VideoCapture(src)
        
        # Set buffer size to 1. We only care about the MOST RECENT frame.
        # If we don't do this, OpenCV might buffer old frames, causing "ghosting" or delay.
        self.capture.set(cv.CAP_PROP_BUFFERSIZE, 1)
        
        self.thread = None
        self.frame = None
        self.running = False
        self.lock = threading.Lock()  # Mutex to prevent reading while writing
        
    def start(self):
        """
        Start the frame reading thread.
        """
        if self.running:
            return
            
        self.running = True
        # Create a daemon thread (dies automatically when main program exits)
        self.thread = threading.Thread(target=self._update_frame)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        """
        Stop the thread and release camera resources.
        """
        self.running = False
        if self.thread is not None:
            self.thread.join() # Wait for thread to finish safely
        self.capture.release()
        
    def _update_frame(self):
        """
        Background loop that continuously reads frames from the camera.
        This runs in the separate thread.
        """
        while self.running:
            ret, frame = self.capture.read()
            if ret:
                # Use lock to ensure we don't write to 'self.frame' 
                # at the exact same moment the main thread is reading it.
                with self.lock:
                    self.frame = frame
            
            # Small sleep to prevent this thread from hogging 100% of a CPU core
            # while spinning in this loop.
            time.sleep(0.01) 
            
    def get_frame(self):
        """
        Get the most recent frame available.
        Returns None if no frame has been read yet.
        """
        with self.lock:
            # Return a copy to prevent race conditions where the frame 
            # might be modified by the update thread while we are processing it.
            return self.frame.copy() if self.frame is not None else None
            
    def is_opened(self):
        """Check if capture source is open."""
        return self.capture.isOpened()