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

video = cv2.VideoCapture("/home/taxen/Desktop/MasterThesis/VideoDescribingLVM/Videos/YSKX3.mp4")

def extract_frames(video_path, fps=3):
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

PROMPT_MESSAGES = """This is a semicolon-separated list of descriptions by annotators watching a video:
            c077 12.10 18.00;c079 11.80 17.30;c080 13.00 18.00;c076 11.80 17.50;c075 5.40 14.10

            the id's mean:
            c075 Tidying up a blanket/s
            c076 Holding a pillow
            c077 Putting a pillow somewhere
            c079 Taking a pillow from somewhere
            c080 Throwing a pillow somewhere 
            
            Write a multi-choice question from this in the format: 
            which action happens between timestamp 1 and timestamp 2?
            A) action 1
            B) action 2
            C) action 4
            etc..

            The timestamps should be one of the timestamps from the annotated video and the alternatives should be the actions in the video.
            Also include the right answer.

            """

PROMPT_MESSAGE_2 =  ["""**Question:**
            Which action happens between timestamp 11.80 and timestamp 17.30?

            A) Tidying up a blanket/s  
            B) Holding a pillow  
            C) Putting a pillow somewhere  
            D) Taking a pillow from somewhere  
            E) Throwing a pillow somewhere  """
            , *map(lambda x: {"image": x, "resize": 768}, Frames),]

params = {
    "model": "gpt-4o",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": PROMPT_MESSAGE_2}
    ],
    "max_tokens": 200,
}

result = client.chat.completions.create(**params)
print(result.choices[0].message.content)