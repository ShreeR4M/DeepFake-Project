import cv2
import mediapipe as mp
import os
import json

from first import extract_frames
from face import extract_face_landmarks


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
    
    # Saving coordinates to JSON file
    with open(output_json, "w") as json_file:
        json.dump(coordinates_output, json_file, indent=4)
    
    return coordinates_output

if __name__ == "__main__":
    video_path = "/home/shree/code/Python/ML_Project/videos/second.mp4"
    output_folder = "frames_output"
    extract_frames(video_path, output_folder)
    coordinates = extract_coordinates_from_frames(output_folder)
