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
                
                # Left eye
                if idx in [33, 133, 157, 158, 159, 160, 161, 173, 246]:
                    face_coordinates["eyes"].append((x, y))
                # Right eye
                elif idx in [362, 263, 386, 387, 388, 466, 390, 373, 374]:
                    face_coordinates["eyes"].append((x, y))
                elif idx in range(1, 18):  # Nose
                    face_coordinates["nose"].append((x, y))
                elif idx in range(61, 89):  # Mouth
                    face_coordinates["mouth"].append((x, y))
                    
        print(f"Found {len(face_coordinates['eyes'])} eye landmarks")
    else:
        print("No face detected!")    
    
    return face_coordinates

def visualize_landmarks(image_path):
    frame = cv2.imread(image_path)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.3)
    
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, landmark in enumerate(face_landmarks.landmark):
                h, w, _ = frame.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                
                # Draw all landmarks in white
                cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)
                
                # Draw eye landmarks in green
                if idx in [33, 133, 157, 158, 159, 160, 161, 173, 246, 362, 263, 386, 387, 388, 466, 390, 373, 374]:
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    
    cv2.imwrite('debug_landmarks.jpg', frame)

visualize_landmarks("/home/shree/code/Python/ML_Project/photos/images.jpg")

def extract_coordinates_from_image(image_path, output_json="image_face_coordinates.json"):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.3,min_tracking_confidence=0.3)
    
    frame = cv2.imread(image_path)
    if frame is None:
        print("Error: Unable to load image.")
        return {}
    
    face_coordinates = extract_face_landmarks(frame, face_mesh)
    
    # Saving coordinates to JSON file
    with open(output_json, "w") as json_file:
        json.dump(face_coordinates, json_file, indent=4)
    
    return face_coordinates

if __name__ == "__main__":
    image_path = "/home/shree/code/Python/ML_Project/photos/images.jpg"
    image_coordinates = extract_coordinates_from_image(image_path)