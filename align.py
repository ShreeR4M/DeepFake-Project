import cv2
import mediapipe as mp
import numpy as np
import os
import json

from face import extract_face_landmarks
from first import extract_frames
from coordinates import extract_coordinates_from_frames



def align_face(reference_image_path, target_frame, face_mesh):
    reference_image = cv2.imread(reference_image_path)
    if reference_image is None:
        print("Error: Unable to load reference image.")
        return None
    
    ref_landmarks = extract_face_landmarks(reference_image, face_mesh)
    tgt_landmarks = extract_face_landmarks(target_frame, face_mesh)
    
    if not ref_landmarks["eyes"] or not tgt_landmarks["eyes"]:
        print("Error: Eyes not detected in either image.")
        return None
    
    # Get all facial landmarks for face region calculation
    ref_points = np.array(ref_landmarks["all"], dtype=np.int32)
    tgt_points = np.array(tgt_landmarks["all"], dtype=np.int32)
    
    # Calculate face bounding box for reference and target
    ref_x, ref_y, ref_w, ref_h = cv2.boundingRect(ref_points)
    tgt_x, tgt_y, tgt_w, tgt_h = cv2.boundingRect(tgt_points)
    
    # Extract eyes for alignment
    ref_left_eye, ref_right_eye = ref_landmarks["eyes"][0], ref_landmarks["eyes"][-1]
    tgt_left_eye, tgt_right_eye = tgt_landmarks["eyes"][0], tgt_landmarks["eyes"][-1]
    
    # Calculate transformation parameters
    ref_eye_center = np.mean([ref_left_eye, ref_right_eye], axis=0)
    tgt_eye_center = np.mean([tgt_left_eye, tgt_right_eye], axis=0)
    
    ref_eye_distance = np.linalg.norm(np.array(ref_right_eye) - np.array(ref_left_eye))
    tgt_eye_distance = np.linalg.norm(np.array(tgt_right_eye) - np.array(tgt_left_eye))
    
    scale = tgt_eye_distance / ref_eye_distance
    angle = np.arctan2(tgt_right_eye[1] - tgt_left_eye[1], tgt_right_eye[0] - tgt_left_eye[0]) - \
            np.arctan2(ref_right_eye[1] - ref_left_eye[1], ref_right_eye[0] - ref_left_eye[0])
    
    # Create transformation matrix to align reference face to target face
    M = cv2.getRotationMatrix2D(tuple(ref_eye_center), np.degrees(angle), scale)
    
    # Add translation to the matrix to move to target face position
    tx = tgt_eye_center[0] - ref_eye_center[0]
    ty = tgt_eye_center[1] - ref_eye_center[1]
    M[0, 2] += tx
    M[1, 2] += ty
    
    # Warp the reference image to align with target
    transformed_ref = cv2.warpAffine(reference_image, M, (target_frame.shape[1], target_frame.shape[0]))
    
    # Create a mask for the warped face region
    mask = np.zeros_like(transformed_ref)
    tgt_hull = cv2.convexHull(tgt_points)
    cv2.fillConvexPoly(mask, tgt_hull, (255, 255, 255))
    
    # Apply color correction to better match skin tones (optional)
    transformed_ref_face = cv2.bitwise_and(transformed_ref, mask)
    
    # Create the result by combining the original frame with the face swap
    result = target_frame.copy()
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask_inv = cv2.bitwise_not(mask_gray)
    
    # Exclude the face area from the original frame
    result_bg = cv2.bitwise_and(result, result, mask=mask_inv)
    
    # Add the warped reference face to the result
    result = cv2.add(result_bg, transformed_ref_face)
    
    # Optional smoothing at the boundaries
    kernel = np.ones((5, 5), np.uint8)
    mask_eroded = cv2.erode(mask_gray, kernel, iterations=2)
    mask_border = mask_gray - mask_eroded   
    result_border = cv2.GaussianBlur(result, (5, 5), 0)
    result = np.where(mask_border[:, :, np.newaxis] > 0, result_border, result)
    
    return result

def align_face_to_video_frames(reference_image_path, frames_folder, output_folder):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.3)

    # Create output folder if not exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for frame_idx in range(1, 10000):
        frame_path = f"{frames_folder}/frame_{frame_idx:04d}.jpg"
        output_path = f"{output_folder}/aligned_frame_{frame_idx:04d}.jpg"
        
        frame = cv2.imread(frame_path)
        if frame is None:
            break
        
        face_swapped_frame = align_face(reference_image_path, frame, face_mesh)
        if face_swapped_frame is None:
            print(f"Skipping frame {frame_idx}, face not detected")
            # Save original frame instead
            cv2.imwrite(output_path, frame)
            continue
        
        cv2.imwrite(output_path, face_swapped_frame)

# Move this into an if __name__ == "__main__" block to prevent execution when imported
if __name__ == "__main__":
    reference_image_path = "/home/shree/code/Python/ML_Project/photos/images.jpg"
    frames_folder = "frames_output"
    aligned_output_folder = "aligned_frames"
    align_face_to_video_frames(reference_image_path, frames_folder, aligned_output_folder)