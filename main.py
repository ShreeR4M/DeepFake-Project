from first import extract_frames, visualize_landmarks
from face import extract_coordinates_from_image
from coordinates import extract_coordinates_from_frames
from align import align_face_to_video_frames

def run_complete_workflow():
    # Paths
    video_path = "/home/shree/code/Python/ML_Project/videos/second.mp4"
    reference_image_path = "/home/shree/code/Python/ML_Project/photos/images.jpg"
    frames_folder = "frames_output"
    aligned_output_folder = "aligned_frames"
    
    extract_frames(video_path, frames_folder)
    
    visualize_landmarks(frames_folder)
    
    ref_coordinates = extract_coordinates_from_image(reference_image_path)
    
    frame_coordinates = extract_coordinates_from_frames(frames_folder)
 
    align_face_to_video_frames(reference_image_path, frames_folder, aligned_output_folder)
    
    print("Complete workflow executed successfully!")

if __name__ == "__main__":
    run_complete_workflow()