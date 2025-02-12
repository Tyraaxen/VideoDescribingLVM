from IPython.display import display, Image, Audio
import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import base64
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
from tqdm import tqdm
from multiprocessing.dummy import Pool


# Load environment variables from the .env file
load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

#==============================================Questions=============================================
event_segmentation = "/home/taxen/VideoDescribingLVM/qvhighlight/filtered_dataset.json"

#====================================================================================================
# Step 1: Load the JSON file with questions
with open(event_segmentation, "r") as f:
    questions_data = json.load(f)

# for event segmentation
def get_query_by_vid(video_id, dataset):
    for item in dataset:
        if item["vid"] == video_id:
            return item["query"]  # Return the corresponding query
    return None  # Return None if the video is not found

#====================================================================================================

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

def model_answers(question_text, video_id, video_path, video_duration): #TODO: add video_duration to the rest of the code

    Frames = extract_frames(video_path)

    PROMPT_MESSAGE = [f"""This is the query: "{question_text}". 
                        Between which timestamps in the video does this event happen?

                        ### **Instructions:**
                        - The video has a total duration of **{video_duration} seconds**.
                        - You **must ensure** that the timestamps you provide are within **0 and {video_duration}**.
                        - Respond in **valid JSON format**, following this exact structure:

                        ```json
                        [
                            {{
                                "vid": "{video_id}",
                                "query": "{question_text}",
                                "answer": [timestamp1, timestamp2]
                            }}
                        ]

                Do not use quotes ("") around the timestamps.
                The timestamps should be in seconds and represent the relative position within the video.
                If the query happens 30 seconds into the video and ends at 120 seconds, the correct response format is:
                "answer": [30, 120]
                Ensure that timestamp1 < timestamp2 and that both are within 0 and {video_duration}
                """, *map(lambda x: {"image": x, "resize": 768}, Frames),]

    params = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are an AI assistant tasked with answering a question based on a set of video frames."},
            {"role": "user", "content": PROMPT_MESSAGE}
        ],
        "max_tokens": 200,
    }

    result = client.chat.completions.create(**params)
    model_output = result.choices[0].message.content


    #======================================================== Create the json file ==================================================================

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
        print(f"Error decoding JSON for video: {e}")
        return []


# ======================================================== Generate Questions for Videos ===========================================================
def process_video(video_id):
    """Function to process each video and return structured results."""
    #video_path = f"/home/taxen/VideoDescribingLVM/Charades_v1_480/{video_id}.mp4"
    video_path = f"/home/taxen/VideoDescribingLVM/qvhighlight/filtered_videos/{video_id}.mp4"

    question_text = get_query_by_vid(video_id, questions_data)

    if question_text:
        return model_answers(question_text, video_id, video_path)  # Expected to return a list of dicts
    return []  # Return an empty list if no question text is found

# Prepare video IDs
video_ids = [entry['vid'] for entry in questions_data[:50]]

# Process all videos in parallel
if __name__ == '__main__':
    num_workers = 1#min(8, len(video_ids))  # Use optimal number of workers
    #print("num_workers: ", 8) #cpu count =8
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(process_video, video_ids), total=len(video_ids), desc="Processing videos", unit="video"))

    # Flatten the list of lists into a single list of dictionaries
    questions = [item for sublist in results for item in sublist]  

    # ====================================================================================================
    # Save to JSON File
    output_path = "/home/taxen/VideoDescribingLVM/answers/openai_answers_es.json"
    with open(output_path, "w") as out_file:
        json.dump(questions, out_file, indent=4)

    print(f"Saved {len(questions)} answers to {output_path}")
