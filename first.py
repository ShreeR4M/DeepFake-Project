import cv2
import mediapipe as mp
import os

def extract_frames(video_path, output_folder):
    # Initialize MediaPipe holistic model (optional, for processing)
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic()
    
    # Creating output folder if not exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Read video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to RGB 
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        holistic.process(frame_rgb)
        
        # Save frame as an image
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {frame_count} frames to {output_folder}")
    return frame_count


def visualize_landmarks(frames_folder, output_folder="visualized_frames"):

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.3)
    
    # Creating output folder if not exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Processing each frame in the folder
    frame_count = 0
    for filename in sorted(os.listdir(frames_folder)):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            frame_path = os.path.join(frames_folder, filename)
            frame = cv2.imread(frame_path)
            
            if frame is None:
                continue
            
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Draw landmarks if face is detected
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
            
            output_path = os.path.join(output_folder, f"visualized_{filename}")
            cv2.imwrite(output_path, frame)


if __name__ == "__main__":
    video_path = "/home/shree/code/Python/ML_Project/videos/second.mp4"
    output_folder = "frames_output"
    extracted_frames = extract_frames(video_path, output_folder)

    visualize_landmarks(output_folder)