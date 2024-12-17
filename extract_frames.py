import cv2
import os

def extract_key_frames(video_path, output_dir, fps=1):
    # Skapa katalog för att spara frames
    os.makedirs(output_dir, exist_ok=True)

    # Läs in videon
    video = cv2.VideoCapture(video_path)
    frame_rate = int(video.get(cv2.CAP_PROP_FPS))
    frame_interval = frame_rate // fps  # Antal frames mellan varje ex
    success, frame = video.read()
    frame_count = 0
    extracted_count = 0

    while success:
        if frame_count % frame_interval == 0:
            frame_name = os.path.join(output_dir, f"frame_{extracted_count}.jpg")
            cv2.imwrite(frame_name, frame)
            extracted_count += 1
        success, frame = video.read()
        frame_count += 1

    video.release()
    print(f"Extracted {extracted_count} frames to {output_dir}")

# Använd funktionen
dir = "C:/Users/tyra_/MasterThesis/VideoDescribingLVM/Frames"
file = "C:/Users/tyra_/MasterThesis/VideoDescribingLVM/file_example.mp4"
extract_key_frames(file, dir, fps=1)
