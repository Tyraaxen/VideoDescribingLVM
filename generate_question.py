'''This code is based on (basically copied from OpenAi cookbook: https://cookbook.openai.com/examples/gpt_with_vision_for_video_understanding)'''
from IPython.display import display, Image, Audio
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import csv
from tqdm import tqdm

# Load environment variables from the .env file
load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

#=============================================== Create dictionairy with video ID and actions =====================================================

video_actions = {}

with open("/home/taxen/Downloads/Charades/Charades_v1_train.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        video_id = row['id']
        video_actions[video_id] = row['actions'].split(';')


#================================================= Read the actions and the corresponding ID =======================================================
# Define the path to your text file
file_path = "/home/taxen/Downloads/Charades/Charades_v1_classes.txt"

# Read the entire file as a string
with open(file_path, "r") as file:
    ACTIONS_TEXT = file.read()

CONTENT = f"""
    You are a QA generating chatbot and your purpose is to create multi-choice questions that will be used to evaluate the video understanding capabilities of LLMs.
    These are the action descriptions for different IDs:
    {ACTIONS_TEXT}
"""

#============================================= Create the prompt to make model generate question ===================================================

def generate_question(vid_ID):
    PROMPT_MESSAGES = f"""This is a semicolon-separated list of descriptions by annotators watching a video:
                {video_actions[vid_ID]}, the video has the ID {vid_ID}
                
                Write a multi-choice question with four alternatives out of this.
                Respond in **valid JSON format**, following this exact structure:

                [
                    {{
                        "video_id": "id",
                        "question": "Which action happens between timestamp X and timestamp Y? A) action 1 B) action 2 C) action 3 D) action 4",
                        "answer": "letter of correct answer"
                    }}
                ]

                The correct answer should be one of the four alternatives. 
                The other 3 alternatives should be chosen among the other actions in the video if there are any. Do not take actions that are very close to the correct action or an action that happens at the same time as the correct action.

                """
    #========================================================= Add prompts to model =================================================================

    params = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": CONTENT},
            {"role": "user", "content": PROMPT_MESSAGES}
        ],
        "max_tokens": 400,
    }

    result = client.chat.completions.create(**params)
    model_output = result.choices[0].message.content

    #======================================================== Create the json file =================================================================

    # Clean JSON output
    if model_output.startswith("```json"):
        model_output = model_output[7:]  # Remove the leading ```json
    if model_output.endswith("```"):
        model_output = model_output[:-3]  # Remove the trailing ```

    # Convert the cleaned string to a valid JSON object
    try:
        parsed_output = json.loads(model_output)  # Convert string to JSON object
        return parsed_output if isinstance(parsed_output, list) else []
    
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON for video {vid_ID}: {e}")
        return []
    
# ======================================================== Generate Questions for Videos ===========================================================
total_videos = len(video_actions)
questions = []

#print("length: ", len(video_actions.keys()))

for ID in tqdm(list(video_actions.keys())[1:1000], desc="Processing videos", unit="video"):
    questions.extend(generate_question(ID))

# =============================================================== Save to JSON File ================================================================
output_path = "/home/taxen/Desktop/MasterThesis/VideoDescribingLVM/Questions/test.json"
with open(output_path, "w") as out_file:
    json.dump(questions, out_file, indent=4)

print(f"Saved {len(questions)} questions to {output_path}")

