import vertexai
from vertexai.language_models import TextGenerationModel
from vertexai.vision_models import Image
import base64
import json
import time
from tqdm import tqdm

# Initialize Vertex AI
vertexai.init(project=os.environ["TT_VERTEXAI_PROJECT"], location="us-central1")
model = TextGenerationModel.from_pretrained("gemini-2.0-flash-exp")

# Load questions
with open("/home/taxen/VideoDescribingLVM/Questions/test.json", "r") as f:
    questions_data = json.load(f)

# Prepare video IDs
video_ids = [entry['video_id'] for entry in questions_data]

def get_question_by_video_id(video_id):
    for entry in questions_data:
        if entry['video_id'] == video_id:
            return entry['question']
    return None  

def encode_image(image_path):
    """Encodes an image as base64 for Vertex AI."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def model_answers(video_id, video_path):
    image_base64 = encode_image(video_path)
    image = Image.from_bytes(base64.b64decode(image_base64))

    question_text = get_question_by_video_id(video_id)
    prompt_text = f"""
    **Question:**
    {question_text}

    Respond in **valid JSON format**, following this exact structure:

        [
            {{
                "video_id": "{video_id}",
                "question": "{question_text}",
                "answer": "letter of your answer"
            }}
        ]
    """

    response = model.predict([prompt_text, image])
    raw_output = response.text if response.text else ""
    
    if not raw_output:
        print(f"\nError: No valid response for video {video_id}.")
        return None

    try:
        cleaned_output = raw_output.replace("```json", "").replace("```", "").strip()
        response_json = json.loads(cleaned_output) 
        return response_json
    except json.JSONDecodeError:
        print(f"\nError: Failed to parse JSON response for video {video_id}.")
        return None

all_answers = []
output_path = "/home/taxen/VideoDescribingLVM/answers/gemini_answers.json"

for video_id in tqdm(video_ids[:100], desc="Processing videos", unit="video"):
    video_path_test = f"/home/taxen/VideoDescribingLVM/Charades_v1_480/{video_id}.mp4"
    result = model_answers(video_id, video_path_test)
    if result:
        all_answers.extend(result)

if all_answers:
    with open(output_path, "w") as out_file:
        json.dump(all_answers, out_file, indent=4)
    print(f"Saved {len(all_answers)} answers to {output_path}")
else:
    print("No valid answers were generated.")
