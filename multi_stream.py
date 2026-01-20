"""
Multiprocessing Stream Handler Module.

This module implements a scalable 'Producer' architecture for video ingestion.
It is designed to decouple video decoding (IO-bound/CPU-bound) from the main
application logic to prevent frame drops and latency.

Key Components:
    - MultiStreamProcessor: Manages a pool of worker processes, one for each RTSP stream.
    - Inter-Process Communication (IPC): Uses `multiprocessing.Queue` (or file buffers)
      to safely transmit detected face crops from workers to the main process.
    - Graceful Shutdown: Handles signal interception (SIGINT) to ensure all child
      processes are terminated correctly.
"""
import multiprocessing as mp
import time
import signal
import sys
from queue import Empty
from stream_processing import stream_worker
import cv2 as cv
import os 

class MultiStreamProcessor:
    def __init__(self, streams, max_queue_size=500 , filtered_images_path = "Quality Assessment\\data\\filtered_aligned_images_quality"):
        self.streams = streams
        self.max_queue_size = max_queue_size
        self.processes = []
        self.face_queue = mp.Queue(maxsize=max_queue_size)
        self.stop_event = mp.Event()
        self.filtered_images_path = filtered_images_path

        # Setup signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        print("\nShutdown signal received. Stopping all processes...")
        self.stop()
        sys.exit(0)
        
    def start(self):
        """Start all stream processing workers"""
        print(f"Starting {len(self.streams)} stream workers...")
        
        for i, stream_url in enumerate(self.streams):
            worker_process = mp.Process(
                target=stream_worker,
                args=(stream_url, self.face_queue, i, self.stop_event),
                name=f"StreamWorker-{i}"
            )
            worker_process.start()
            self.processes.append(worker_process)
            print(f"Started worker {i} for stream: {stream_url}")
            
        print("All stream workers started successfully")
        
    def get_faces_batch(self, max_batch_size=50, timeout=1.0):
        """
        Get a batch of faces from the queue folder
        Returns list of face data dictionaries
        """
        faces_batch = []
        start_time = time.time()
        
        while (len(faces_batch) < max_batch_size and 
               time.time() - start_time < timeout):
            try:
                for filename in os.listdir(self.filtered_images_path):
                    if filename.endswith(".jpg"):
                        img_path = os.path.join(self.filtered_images_path,filename)
                        img = cv.imread(img_path)
                        parts = filename.replace('.jpg', '').split("_")

                    face_data = {
                                'image': img,
                                "face_quality":float(parts[1]),
                                'face_idx': int(parts[3]),
                                'frame_timestamp': parts[4],
                                'frame_count': int(parts[5]),
                                'worker_id': int(parts[6])
                                }
                    faces_batch.append(face_data)
                    os.remove(img_path)
            except:
                break
                
        return faces_batch
        
    def get_queue_size(self):
        """Get current queue size"""
        return len(os.listdir(self.filtered_images_path))
        
    def stop(self):
        """Stop all worker processes"""
        print("Stopping all stream workers...")
        
        # Signal all workers to stop
        self.stop_event.set()
        
        # Wait for processes to finish (with timeout)
        for i, process in enumerate(self.processes):
            process.join(timeout=5.0)
            if process.is_alive():
                print(f"Force terminating worker {i}")
                process.terminate()
                process.join()
                
        # Clear the queue
        while not self.face_queue.empty():
            try:
                self.face_queue.get_nowait()
            except:
                break
                
        print("All workers stopped")