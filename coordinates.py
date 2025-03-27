import cv2
import mediapipe as mp
import os
import json

from first import extract_frames

# def extract_frames(video_path, output_folder):
#     # Initialize MediaPipe holistic model (optional, for processing)
#     mp_holistic = mp.solutions.holistic
#     holistic = mp_holistic.Holistic()
    
#     # Create output folder if not exists
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
    
#     # Read video
#     cap = cv2.VideoCapture(video_path)
#     frame_count = 0
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # Convert frame to RGB (MediaPipe expects RGB images)
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#         # Process with MediaPipe (optional, e.g., for pose detection)
#         results = holistic.process(frame_rgb)
        
#         # Save frame as an image
#         frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
#         cv2.imwrite(frame_filename, frame)
        
#         frame_count += 1
    
#     cap.release()
#     print(f"Extracted {frame_count} frames to {output_folder}")

def extract_face_landmarks(frame, face_mesh):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    
    face_coordinates = {
        "all": [],
        "eyes": [],
        "nose": [],
        "mouth": []
    }
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, landmark in enumerate(face_landmarks.landmark):
                h, w, _ = frame.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                face_coordinates["all"].append((x, y))
                
                # Categorize landmarks based on MediaPipe FaceMesh landmark indices
                if idx in range(33, 133):  # Eyes
                    face_coordinates["eyes"].append((x, y))
                elif idx in range(1, 18):  # Nose
                    face_coordinates["nose"].append((x, y))
                elif idx in range(61, 89):  # Mouth
                    face_coordinates["mouth"].append((x, y))
    
    return face_coordinates

def extract_coordinates_from_frames(output_folder, output_json="face_coordinates.json"):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()
    
    coordinates_output = {}
    
    for filename in sorted(os.listdir(output_folder)):
        if filename.endswith(".jpg"):
            frame_path = os.path.join(output_folder, filename)
            frame = cv2.imread(frame_path)
            face_coordinates = extract_face_landmarks(frame, face_mesh)
            coordinates_output[filename] = face_coordinates
    
    # Save coordinates to JSON file
    with open(output_json, "w") as json_file:
        json.dump(coordinates_output, json_file, indent=4)
    
    return coordinates_output

# Example usage
video_path = "input_video/home/shree/code/Python/ML_Project/videos/second.mp4"
output_folder = "frames_output"
extract_frames(video_path, output_folder)
coordinates = extract_coordinates_from_frames(output_folder)
print("Extracted face coordinates saved to face_coordinates.json")
