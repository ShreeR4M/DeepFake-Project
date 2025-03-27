import cv2
import mediapipe as mp
import json

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

def extract_coordinates_from_image(image_path, output_json="image_face_coordinates.json"):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()
    
    frame = cv2.imread(image_path)
    if frame is None:
        print("Error: Unable to load image.")
        return {}
    
    face_coordinates = extract_face_landmarks(frame, face_mesh)
    
    # Save coordinates to JSON file
    with open(output_json, "w") as json_file:
        json.dump(face_coordinates, json_file, indent=4)
    
    print(f"Extracted face coordinates saved to {output_json}")
    return face_coordinates

# Example usage
image_path = "/home/shree/code/Python/ML_Project/photos/images.jpg"
image_coordinates = extract_coordinates_from_image(image_path)
print("Extracted face coordinates saved to image_face_coordinates.json")
