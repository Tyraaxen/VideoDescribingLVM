import os
import json
import torch
import av
import numpy as np
from tqdm import tqdm

# -----------------------------
# 1) Import Video-LLaVA classes
# -----------------------------
# Adapt these imports to your actual model/processor:
from transformers import (
    VideoLlavaForConditionalGeneration,
    VideoLlavaProcessor
)

# -------------------------------------------------
# 2) Function to read frames from a video with PyAV
# -------------------------------------------------
def read_video_pyav(container, indices):
    """
    Decode the video with PyAV decoder.
    Args:
        container (av.container.input.InputContainer): PyAV container.
        indices (List[int]): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, H, W, 3).
    """
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

# -------------------------------------------------
# 3) Load the question data from JSON
# -------------------------------------------------
QUESTION_FILE = "/home/taxen/VideoDescribingLVM/Questions/test.json"
with open(QUESTION_FILE, "r") as f:
    questions_data = json.load(f)

def get_question_by_video_id(video_id):
    for entry in questions_data:
        if entry['video_id'] == video_id:
            return entry['question']
    return None

# -------------------------------------------------
# 4) Define the inference function
# -------------------------------------------------
def model_answers(model, processor, question_text, video_id, video_path):
    """
    Use the loaded Video-LLaVA model & processor to generate a JSON answer.
    """
    # 1) Read the video frames
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    # Sample 8 frames uniformly
    indices = np.linspace(0, total_frames - 1, num=16, dtype=int)
    video = read_video_pyav(container, indices)

    # 2) Construct the prompt
    prompt = (f"<video> You are a helpful AI. The question is: {question_text}\n"
    "Answer with exactly one letter (A, B, C, or D).")

    # 3) Generate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = processor(text=prompt, videos=video, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200)

    raw_output = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    return raw_output
    # 4) Post-process: attempt to isolate valid JSON
    """start_idx = raw_output.find("[")
    end_idx = raw_output.rfind("]") + 1
    json_str = raw_output[start_idx:end_idx].strip() if (start_idx != -1 and end_idx != -1) else "[]"
    print("json string", json_str, "json string ended")
    # 5) Parse the JSON
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        parsed = []
    return parsed if isinstance(parsed, list) else []"""

# -------------------------------------------------
# 5) Main script without multiprocessing
# -------------------------------------------------
if __name__ == '__main__':
    # 1) Load the model & processor once
    model = VideoLlavaForConditionalGeneration.from_pretrained(
        "LanguageBind/Video-LLaVA-7B-hf",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")

    video_ids = [entry['video_id'] for entry in questions_data[:10]]
    results = []

    # 2) Process each video in a simple for-loop
    for video_id in tqdm(video_ids, desc="Processing videos", unit="video"):
        video_path = f"/home/taxen/VideoDescribingLVM/Charades_v1_480/{video_id}.mp4"
        question_text = get_question_by_video_id(video_id)
        if question_text:
            answers = model_answers(model, processor, question_text, video_id, video_path)
            print("answers: ", answers)
            results.extend(answers)

    # 3) Save final results to JSON
    OUTPUT_JSON = "/home/taxen/VideoDescribingLVM/Questions/test_answers.json"
    with open(OUTPUT_JSON, "w") as out_file:
        json.dump(results, out_file, indent=4)

    print(f"Saved {len(results)} answers to {OUTPUT_JSON}")