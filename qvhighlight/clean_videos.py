import json
import os
import shutil

# Define file paths
json_file = "qvhighlight/filtered_dataset.json"  # JSON file with filtered objects
video_folder = "qvhilights_videos/videos"  # Folder containing all videos
output_folder = "qvhilights_videos/filtered_videos"  # Folder where relevant videos will be saved

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load the filtered dataset
with open(json_file, "r", encoding="utf-8") as f:
    filtered_data = json.load(f)

# Extract the relevant video filenames
video_filenames = {obj["vid"] + ".mp4" for obj in filtered_data}  # Add .mp4 extension

# Process the videos
for filename in os.listdir(video_folder):
    if filename in video_filenames:
        # Copy the relevant video to the output folder
        shutil.copy(os.path.join(video_folder, filename), os.path.join(output_folder, filename))
        print(f"Copied: {filename}")

print(f"Filtered videos saved to {output_folder}. Total videos kept: {len(video_filenames)}")
