import cv2
import mediapipe as mp
import os

def extract_frames(video_path, output_folder):
    # Initialize MediaPipe holistic model (optional, for processing)
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic()
    
    # Create output folder if not exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Read video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to RGB (MediaPipe expects RGB images)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe (optional, e.g., for pose detection)
        results = holistic.process(frame_rgb)
        
        # Save frame as an image
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {frame_count} frames to {output_folder}")

# # Example usage
# video_path = "/home/shree/code/Python/ML_Project/videos/second.mp4"
# output_folder = "frames_output"
# extract_frames(video_path, output_folder)

if __name__ == "__main__":
    video_path = "/home/shree/code/Python/ML_Project/videos/second.mp4"
    output_folder = "frames_output"
    extract_frames(video_path, output_folder)