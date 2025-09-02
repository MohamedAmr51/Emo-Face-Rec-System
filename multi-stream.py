from stream_processing import start_stream

# Omit input to call default camera
streams_G8_zone = [
"rtsp://admin:LV@0000@lv@10.10.10.134/onvif/profile2/media.smp",            #Management Floor Cam 1
"rtsp://admin:LV@0000@lv@10.10.10.135/onvif/profile2/media.smp"             #Management Floor Cam 2
# "rtsp://admin:V90@13579@v90@10.30.10.127/onvif/profile2/media.smp",       #Waiting Area 90
# "rtsp://admin:V90@0000@v90@10.30.10.103/onvif/profile2/media.smp",        #Customer Service 90
# "rtsp://admin:V90@0000@v90@10.30.10.111/onvif/profile5/media.smp",        #The safe 2
# "rtsp://admin:V90@0000@v90@10.30.10.117/onvif/profile2/media.smp",        #Entrance Corridor
# "rtsp://admin:admin@13579@10.10.10.67:554/onvif/profile2/media.smp",      #Garden 8 
# "rtsp://admin:V90@0000@v90@10.30.10.118/onvif/profile2/media.smp"         #Meeting Room 3
]

for i , stream in enumerate(streams_G8_zone):
    print(f"Starting Stream : {i+1}")
    start_stream(stream)