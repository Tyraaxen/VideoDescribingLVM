'''This code is based on (basically copied from OpenAi cookbook: https://cookbook.openai.com/examples/gpt_with_vision_for_video_understanding)'''
from IPython.display import display, Image, Audio
import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import base64
import time
from openai import OpenAI
import os
import requests
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

video = cv2.VideoCapture("/home/taxen/Master Thesis/VideoDescribingLVM/tyst_minut.mp4")

def extract_frames(video_path, fps=1):
    #Extract frames and make into base64 format
    frame_rate = int(video_path.get(cv2.CAP_PROP_FPS))
    frame_interval = frame_rate // fps  # Antal frames mellan varje ex
    success, frame = video.read()
    frame_count = 0
    base64Frames = []

    while success:
        if frame_count % frame_interval == 0:
            #if not success:
                #break
            _, buffer = cv2.imencode(".jpg", frame)
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        success, frame = video.read()
        frame_count += 1

    video.release()
    print(len(base64Frames), "frames read.")
    return base64Frames

Frames = extract_frames(video)

PROMPT_MESSAGES = [
    {
        "role": "user",
        "content": [
            "These are frames from a video. What is happening in the video? Is these any part of the video that is not so interesting?",
            *map(lambda x: {"image": x, "resize": 768}, Frames),
        ],
    },
]
params = {
    "model": "gpt-4o",
    "messages": PROMPT_MESSAGES,
    "max_tokens": 200,
}

result = client.chat.completions.create(**params)
print(result.choices[0].message.content)