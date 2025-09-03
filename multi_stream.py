import multiprocessing as mp
import time
import signal
import sys
from queue import Empty
from stream_processing import stream_worker

# # Omit input to call default camera
# streams_G8_zone = [
# "rtsp://admin:LV@0000@lv@10.10.10.134/onvif/profile2/media.smp",            #Management Floor Cam 1
# "rtsp://admin:LV@0000@lv@10.10.10.135/onvif/profile2/media.smp"             #Management Floor Cam 2
# # "rtsp://admin:V90@13579@v90@10.30.10.127/onvif/profile2/media.smp",       #Waiting Area 90
# # "rtsp://admin:V90@0000@v90@10.30.10.103/onvif/profile2/media.smp",        #Customer Service 90
# # "rtsp://admin:V90@0000@v90@10.30.10.111/onvif/profile5/media.smp",        #The safe 2
# # "rtsp://admin:V90@0000@v90@10.30.10.117/onvif/profile2/media.smp",        #Entrance Corridor
# # "rtsp://admin:admin@13579@10.10.10.67:554/onvif/profile2/media.smp",      #Garden 8 
# # "rtsp://admin:V90@0000@v90@10.30.10.118/onvif/profile2/media.smp"         #Meeting Room 3
# ]

class MultiStreamProcessor:
    def __init__(self, streams, max_queue_size=500):
        self.streams = streams
        self.max_queue_size = max_queue_size
        self.processes = []
        self.face_queue = mp.Queue(maxsize=max_queue_size)
        self.stop_event = mp.Event()
        
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
        Get a batch of faces from the queue
        Returns list of face data dictionaries
        """
        faces_batch = []
        start_time = time.time()
        
        while (len(faces_batch) < max_batch_size and 
               time.time() - start_time < timeout):
            try:
                face_data = self.face_queue.get(timeout=0.1)
                faces_batch.append(face_data)
            except:
                break
                
        return faces_batch
        
    def get_queue_size(self):
        """Get current queue size"""
        return self.face_queue.qsize()
        
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