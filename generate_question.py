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
            
            Write a multi-choice question from this in the format: 
            which action happens between timestamp 1 and timestamp 2?
            A) action 1
            B) action 2
            C) action 4
            etc..

            The timestamps should be one of the timestamps from the annotated video and the alternatives should be the actions in the video.
            Also include the right answer.

            """

CONTENT = """
    You are a helpful assistant.
    These are the action descriptions of different IDs:
c000 Holding some clothes
c001 Putting clothes somewhere
c002 Taking some clothes from somewhere
c003 Throwing clothes somewhere
c004 Tidying some clothes
c005 Washing some clothes
c006 Closing a door
c007 Fixing a door
c008 Opening a door
c009 Putting something on a table
c010 Sitting on a table
c011 Sitting at a table
c012 Tidying up a table
c013 Washing a table
c014 Working at a table
c015 Holding a phone/camera
c016 Playing with a phone/camera
c017 Putting a phone/camera somewhere
c018 Taking a phone/camera from somewhere
c019 Talking on a phone/camera
c020 Holding a bag
c021 Opening a bag
c022 Putting a bag somewhere
c023 Taking a bag from somewhere
c024 Throwing a bag somewhere
c025 Closing a book
c026 Holding a book
c027 Opening a book
c028 Putting a book somewhere
c029 Smiling at a book
c030 Taking a book from somewhere
c031 Throwing a book somewhere
c032 Watching/Reading/Looking at a book
c033 Holding a towel/s
c034 Putting a towel/s somewhere
c035 Taking a towel/s from somewhere
c036 Throwing a towel/s somewhere
c037 Tidying up a towel/s
c038 Washing something with a towel
c039 Closing a box
c040 Holding a box
c041 Opening a box
c042 Putting a box somewhere
c043 Taking a box from somewhere
c044 Taking something from a box
c045 Throwing a box somewhere
c046 Closing a laptop
c047 Holding a laptop
c048 Opening a laptop
c049 Putting a laptop somewhere
c050 Taking a laptop from somewhere
c051 Watching a laptop or something on a laptop
c052 Working/Playing on a laptop
c053 Holding a shoe/shoes
c054 Putting shoes somewhere
c055 Putting on shoe/shoes
c056 Taking shoes from somewhere
c057 Taking off some shoes
c058 Throwing shoes somewhere
c059 Sitting in a chair
c060 Standing on a chair
c061 Holding some food
c062 Putting some food somewhere
c063 Taking food from somewhere
c064 Throwing food somewhere
c065 Eating a sandwich
c066 Making a sandwich
c067 Holding a sandwich
c068 Putting a sandwich somewhere
c069 Taking a sandwich from somewhere
c070 Holding a blanket
c071 Putting a blanket somewhere
c072 Snuggling with a blanket
c073 Taking a blanket from somewhere
c074 Throwing a blanket somewhere
c075 Tidying up a blanket/s
c076 Holding a pillow
c077 Putting a pillow somewhere
c078 Snuggling with a pillow
c079 Taking a pillow from somewhere
c080 Throwing a pillow somewhere
c081 Putting something on a shelf
c082 Tidying a shelf or something on a shelf
c083 Reaching for and grabbing a picture
c084 Holding a picture
c085 Laughing at a picture
c086 Putting a picture somewhere
c087 Taking a picture of something
c088 Watching/looking at a picture
c089 Closing a window
c090 Opening a window
c091 Washing a window
c092 Watching/Looking outside of a window
c093 Holding a mirror
c094 Smiling in a mirror
c095 Washing a mirror
c096 Watching something/someone/themselves in a mirror
c097 Walking through a doorway
c098 Holding a broom
c099 Putting a broom somewhere
c100 Taking a broom from somewhere
c101 Throwing a broom somewhere
c102 Tidying up with a broom
c103 Fixing a light
c104 Turning on a light
c105 Turning off a light
c106 Drinking from a cup/glass/bottle
c107 Holding a cup/glass/bottle of something
c108 Pouring something into a cup/glass/bottle
c109 Putting a cup/glass/bottle somewhere
c110 Taking a cup/glass/bottle from somewhere
c111 Washing a cup/glass/bottle
c112 Closing a closet/cabinet
c113 Opening a closet/cabinet
c114 Tidying up a closet/cabinet
c115 Someone is holding a paper/notebook
c116 Putting their paper/notebook somewhere
c117 Taking paper/notebook from somewhere
c118 Holding a dish
c119 Putting a dish/es somewhere
c120 Taking a dish/es from somewhere
c121 Wash a dish/dishes
c122 Lying on a sofa/couch
c123 Sitting on sofa/couch
c124 Lying on the floor
c125 Sitting on the floor
c126 Throwing something on the floor
c127 Tidying something on the floor
c128 Holding some medicine
c129 Taking/consuming some medicine
c130 Putting groceries somewhere
c131 Laughing at television
c132 Watching television
c133 Someone is awakening in bed
c134 Lying on a bed
c135 Sitting in a bed
c136 Fixing a vacuum
c137 Holding a vacuum
c138 Taking a vacuum from somewhere
c139 Washing their hands
c140 Fixing a doorknob
c141 Grasping onto a doorknob
c142 Closing a refrigerator
c143 Opening a refrigerator
c144 Fixing their hair
c145 Working on paper/notebook
c146 Someone is awakening somewhere
c147 Someone is cooking something
c148 Someone is dressing
c149 Someone is laughing
c150 Someone is running somewhere
c151 Someone is going from standing to sitting
c152 Someone is smiling
c153 Someone is sneezing
c154 Someone is standing up from somewhere
c155 Someone is undressing
c156 Someone is eating something 
"""


params = {
    "model": "gpt-4o",
    "messages": [
        {"role": "system", "content": CONTENT},
        {"role": "user", "content": PROMPT_MESSAGES}
    ],
    "max_tokens": 200,
}

result = client.chat.completions.create(**params)
print(result.choices[0].message.content)