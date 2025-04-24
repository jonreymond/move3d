import cv2
import os
import json
from rtmlib import Body, draw_skeleton

import hydra
from omegaconf import DictConfig


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
        keypoints (_type_): _description_
        scores (_type_): _description_
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
    return save_2d_to_json(filename, keypoints=keypoints, scores=scores, output_json_path=output_json_path)
    
    




@hydra.main(version_base=None, 
            # config_path="../configs", 
            config_path=os.path.join(os.path.dirname(__file__), "../configs"),
            config_name="estimation_2d")
def main(config:DictConfig):
    
    model = Body(
        mode=config.model.mode,
        backend=config.model.backend,
        device=config.model.device)
    
    for filename in os.listdir(config.video_folder):
        if filename.endswith('.MP4'):
            video_path = os.path.join(config.video_folder, filename)
            output_video_path = os.path.join(config.output_folder, f'keypoints_{filename}')
            output_json_path = os.path.join(config.output_json_folder, f'keypoints_{os.path.splitext(filename)[0]}.json')

            print(f"Processing: {filename}")
            process_2D(model, video_path, output_video_path, output_json_path)
            
    cv2.destroyAllWindows()
    



if __name__ == "__main__":
    main()
    print('done')