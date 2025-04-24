import cv2
import os
import json
from rtmlib import Body, draw_skeleton

import hydra
from omegaconf import DictConfig
from motion_BERT import *

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np

def get_current_wd():
    print(f"Current working directory: {os.getcwd()}")



def get_video_writer(video_path, output_video_path):
    """create a video writer 

    Args:
        video_path (str): path of the current video
        output_video_path (str): path of the video that we want to create

    Returns:
        (VideoCapture, VideoWriter): 
    """
    cap = cv2.VideoCapture(video_path)

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)

    print(f"  FPS: {fps}, Frame Size: {frame_size}")
    if fps == 0 or width == 0 or height == 0:
        print(f"Skipping {video_path} due to invalid FPS or frame size.")
        cap.release()
        return None, None 
    return cap, cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)
    
 
 
    
def save_2d_to_json(filename, keypoints, scores, output_json_path):
    """save

    Args:
        filename (_type_): _description_
        keypoints (np.ndarray): _description_
        scores (np.ndarray): _description_
        output_json_path (_type_): _description_
    
    Returns:
        Dict: json_data
    """
    # Save keypoints to JSON
    
    
    json_data = {
        "video": filename,
        "keypoints": keypoints,
        "scores": scores
    }

    with open(output_json_path, 'w') as f:
        json.dump(json_data, f)

    print(f"Saved keypoints JSON: {output_json_path}")
    return json_data



def process_2D(model, video_path, output_video_path, output_json_path):
    """from a 2D model from RTM, generate keypoints from the given video, 
    and store the keypoints in a json format, and generate a video with the keypoints.

    Args:
        model (rtmlib.Body): the rtmlib.Body 2D pose model
        video_path (str): path of the video we want to generate the keypoints
        output_video_path (str): path were we want to store the resulted video
        output_json_path (_type_): path were we want to store the keypoints in a JSON format

    Returns:
        Dict: return the JSON data as a dict
    """
    cap, out = get_video_writer(video_path, output_video_path)
    if not cap:
        print('error')
        return None
    
    all_keypoints = []
    all_scores = []

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        keypoints, scores = model(frame)
        all_keypoints.append(keypoints.tolist())
        all_scores.append(scores.tolist())

        frame = draw_skeleton(frame, keypoints, scores, kpt_thr=0.5)
        out.write(frame)

        frame_count += 1
        if frame_count % 200 == 0:
            print(f"  Processed {frame_count} frames...")

    cap.release()
    out.release()
    print(f"Saved video: {output_video_path}")

    if frame_count == 0:
        print(f"No frames were written to {output_video_path}") 
    else:
        print(f"Saved {frame_count} frames to {output_video_path}")
        
    filename = os.path.basename(video_path)
    return save_2d_to_json(filename, keypoints=all_keypoints, scores=all_scores, output_json_path=output_json_path)
    
def detect_patient(video_path, output_json_path):
    """
    Ask user to select a person on frame 0, then track that person across frames.
    """
    with open(output_json_path, "r") as f:
        raw_data = json.load(f)

    frame0 = raw_data["keypoints"][0]
    scores0 = raw_data["scores"][0]

    if isinstance(frame0[0][0], list):  # Multiple people
        people_kpts = frame0
        print(f"→ Number of people in frame 0: {len(frame0)}")
        print('Multiple people detected on the video. Chose the ID of the person of interest.')
        print('Please refer to the ID in the legend of the picture.')

        plot(people_kpts)  # you already have this
        while True:
            try:
                user_input = int(input("Please enter the ID as an integer: "))
                if 0 <= user_input < len(frame0):
                    print(f"You selected: {user_input}")
                    break
                else:
                    print("Number not in range. Try again.")
            except ValueError:
                print("Invalid input. Please enter an integer.")

        # Reference keypoints to track
        ref_kpts = np.array(frame0[user_input])

        extracted_keypoints = []
        extracted_scores = []

        for frame_kpts, frame_scores in zip(raw_data["keypoints"], raw_data["scores"]):
            frame_kpts_np = np.array(frame_kpts)  # shape (N_people, N_keypoints, 2)
            frame_scores_np = np.array(frame_scores)

            if len(frame_kpts_np.shape) == 3:  # still multiple people
                # Compute distance between each detected person and the reference
                dists = np.linalg.norm(frame_kpts_np - ref_kpts, axis=(1, 2))  # (N_people,)
                best_idx = np.argmin(dists)
                extracted_keypoints.append(frame_kpts[best_idx])
                extracted_scores.append(frame_scores[best_idx])
            else:
                # Only one person
                extracted_keypoints.append(frame_kpts)
                extracted_scores.append(frame_scores)

        new_data = {
            "video": raw_data["video"],
            "keypoints": extracted_keypoints,
            "scores": extracted_scores
        }

        plot(new_data["keypoints"][0], name_fig='Selected_participant.png')
        return new_data

    else:
        print("Only one person detected.")
        return raw_data

def detect_patient_old(video_path, output_json_path):
    """
    Script does: 
    - detect if several people
    - plot the keypoints in different colors + first image
    - ask the user which person they want to keep
    - return the corresponding data

    """   
    # Load the json
    with open(output_json_path, "r") as f:
        raw_data = json.load(f)
    
    # Get first frame
    frame_kpts = raw_data["keypoints"][0]

    # Handle single/multiple person
    if isinstance(frame_kpts[0][0], list):
        people_kpts = frame_kpts  # multiple people
        print(f"→ Number of people in frame 0: {len(frame_kpts)}")
        print('Multiple people detected on the video. Chose the ID of the person of interest.')
        print('Please refer to the ID in the legend of the picture.')
        
        plot(people_kpts)
        while True:
            try:
                user_input = int(input("Please enter the ID as an integer: "))
                if 0 <= user_input <= (len(frame_kpts)-1):
                    print(f"You selected: {user_input}")
                    break
                else:
                    print("Number not in range. Try again.")
            except ValueError:
                print("Invalid input. Please enter an integer.")


        extracted_keypoints = [
            frame[user_input] for frame in raw_data["keypoints"]
        ]

        extracted_scores = [
            frame[user_input] for frame in raw_data["scores"]
        ]

        new_data = {
            "video": raw_data["video"],
            "keypoints": extracted_keypoints,
            "scores": extracted_scores
        }
        # To plot the first frame 
        keypoints = new_data["keypoints"][0] 
        
        plot(keypoints, name_fig='Selected_participant.png')

        return new_data
    
    
    else:
        people_kpts = [frame_kpts]  # single person
        print('Only one person detected on the video.')
        return raw_data




def plot(frame_kpts, name_fig="Choose_participant.png"):
    """
    Plot keypoints for the first frame
    """
    # Handle single/multiple person
    if isinstance(frame_kpts[0][0], list):
        people_kpts = frame_kpts  # multiple people
    else:
        people_kpts = [frame_kpts]  # single person

    # COCO skeleton (17 joints)
    skeleton = [
        (5, 7), (7, 9),       # Left arm
        (6, 8), (8,10),       # Right arm
        (11,13), (13,15),     # Left leg
        (12,14), (14,16),     # Right leg
        (5,6), (11,12),       # Shoulders & hips
        (0,1), (1,3), (0,2), (2,4),  # Nose → eyes → ears
        (5,11), (6,12)        # Torso links
    ]

    # Use colormap for people
    num_people = len(people_kpts)
    colors = cm.get_cmap('tab10', num_people)

    plt.figure(figsize=(10, 7))

    for idx, kpts in enumerate(people_kpts):
        xs = [pt[0] for pt in kpts]
        ys = [pt[1] for pt in kpts]
        plt.scatter(xs, ys, label=f"Person {idx}", color=colors(idx), s=40)

        for i, j in skeleton:
            if i < len(kpts) and j < len(kpts):
                x1, y1 = kpts[i]
                x2, y2 = kpts[j]
                plt.plot([x1, x2], [y1, y2], color=colors(idx), linewidth=2)

    plt.title("First frame - All people (Color-coded)")
    plt.legend()
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.savefig(name_fig)
    plt.show()




@hydra.main(version_base=None, 
            # config_path="../configs", 
            config_path=os.path.join(os.path.dirname(__file__), "../configs"),
            config_name="estimation_2d")
def main(config:DictConfig):
    
    model = Body(
        mode=config.model.mode,
        backend=config.model.backend,
        device=config.model.device)
    print(f"Looking for videos in: {config.video_folder}")
    print("Files found:", os.listdir(config.video_folder))
    
    for filename in os.listdir(config.video_folder):
        if (filename.endswith('.MP4') or filename.endswith('.mp4')):
            video_path = os.path.join(config.video_folder, filename)
            output_video_path = os.path.join(config.output_folder, f'keypoints_{filename}')
            output_json_path = os.path.join(config.output_json_folder, f'keypoints_{os.path.splitext(filename)[0]}.json')
            alpha_pose_output_path = os.path.join(config.output_json_folder, f'alpha_format_keypoints_{os.path.splitext(filename)[0]}.json')
            video_3d_path = os.path.join(config.video_3d_path, f'keypoints_{filename}')

            if not os.path.exists(video_3d_path):
                os.makedirs(video_3d_path)
            
            print(f"Processing: {filename}")
            process_2D(model, video_path, output_video_path, output_json_path)
            raw_data = detect_patient(video_path, output_json_path)
            
            convert_JSON_MB_format(raw_data, alpha_pose_output_path)
            motion_BERT(alpha_pose_output_path, video_path, video_3d_path)

            show_video(video_3d_path)
            
    cv2.destroyAllWindows()
    



if __name__ == "__main__":
    main()
    print('done')