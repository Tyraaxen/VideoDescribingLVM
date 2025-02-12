import av
import torch
import numpy as np
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor
from huggingface_hub import hf_hub_download
import json
import base64

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (av.container.input.InputContainer): PyAV container.
        indices (List[int]): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
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


def model_answers(question_text, video_id, video_path):
    # Load the model in half-precision
    model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf", torch_dtype=torch.float16, device_map="auto")
    processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")

    # Load the video as an np.array, sampling uniformly 8 frames
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / 8).astype(int)
    video = read_video_pyav(container, indices)

    # For better results, we recommend to prompt the model in the following format
    prompt = f"""
    <video>

    Respond **only** with valid JSON (no extra text, no markdown). 
    Use this exact structure and fill the answer field with the correct letter:

    [
        {{
            "video_id": "{video_id}",
            "question": "{question_text}",
            "answer": "letter of your answer"
        }}
    ]
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = processor(text=prompt, videos=video, return_tensors="pt").to(device)

    out = model.generate(**inputs, max_new_tokens=200)
    printout = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return printout

with open("/home/taxen/VideoDescribingLVM/Questions/test.json", "r") as f:
    questions_data = json.load(f)

def get_question_by_video_id(video_id):
    for entry in questions_data:
        if entry['video_id'] == video_id:
            return entry['question']
    return None  # Return None if the video_id is not found

vid_id = "N11GT"
path =f"/home/taxen/VideoDescribingLVM/Charades_v1_480/{vid_id}.mp4"
question = get_question_by_video_id(vid_id)

print(model_answers(question, vid_id, path))