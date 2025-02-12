import cv2
import os
import base64
import requests

def extract_frames(video_path, fps=1):
    #Extract frames and make into base64 format
    video_id = cv2.VideoCapture(video_path)
    frame_rate = int(video_id.get(cv2.CAP_PROP_FPS))
    frame_interval = frame_rate // fps  # Antal frames mellan varje ex
    success, frame = video_id.read()
    frame_count = 0
    base64Frames = []

    while success:
        if frame_count % frame_interval == 0:
            #if not success:
                #break
            _, buffer = cv2.imencode(".jpg", frame)
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        success, frame = video_id.read()
        frame_count += 1

    video_id.release()
    #print(frame_count, "frames read.")
    print(len(base64Frames), "frames read.")
    return base64Frames
