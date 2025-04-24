import json
import os

def motion_BERT():

    # Convert JSON to MotionBERT compatible files
    json_to_convert_file_path = [] # TO COMPLETE
    converted_json_path = [] # TO COMPLETE
    
    convert_JSON_MB_format(json_to_convert_file_path, converted_json_path)

    # Run MotionBERT function with pretrained model
    video_path = []
    config_path = os.path.join('..', '..', 'MotionBERT', 'configs', 'pose3d', 'MB_ft_h36m_global_lite.yaml')
    eval_path = os.path.join('..', '..', 'MotionBERT', 'checkpoint', 'pose3d', 'FT_MB_lite_MB_ft_h36m_global_lite', 'best_epoch.bin')
    output_path = os.path.join('..', 'data', 'reconstruction')
    infer_script = os.path.join("..", "..", "MotionBERT", "infer_wild.py")

    # Run MotionBERT command
    # python MotionBERT/infer_wild.py  
    # --vid_path /pvc/scratch/3dmotion/20250326/camera/C0110.MP4   
    # --json_path data/keypoints/keypoints_C0110_converted.json   
    # --out_path data/reconstruction/   
    # --config MotionBERT/configs/pose3d/MB_ft_h36m_global_lite.yaml  
    # --evaluate MotionBERT/checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin   
    # --pixel
    
    command = [
        "python", infer_script,
        "--vid_path", video_path,
        "--json_path", converted_json_path,
        "--out_path", output_path,
        "--config", config_path,
        "--evaluate", eval_path,
        "--pixel"
    ]

    # Execute
    subprocess.run(command, check=True)



def convert_JSON_MB_format(json_file_path, output_path):
    # Load the JSON your friend exported
    with open(json_file_path, "r") as f:
        raw_data = json.load(f)

    all_keypoints = raw_data["keypoints"]  # List of [17 x 2] keypoints per frame
    all_scores = raw_data["scores"]        # List of [17 x 1] scores per frame

    alphapose_format = []

    for idx, (frame_kpts, frame_scores) in enumerate(zip(all_keypoints, all_scores)):
        # Reconstruct [x, y, score] for each keypoint
        frame_combined = []
        for (xy, s) in zip(frame_kpts, frame_scores):
            frame_combined.extend([xy[0], xy[1], s])
        avg_score = sum(s[0] if isinstance(s, list) else s for s in frame_scores) / len(frame_scores)

        entry = {
            "image_id": idx,
            "keypoints": frame_combined,  # list of 51 floats
            "score": float(avg_score)  # average frame confidence
        }
        alphapose_format.append(entry)

    # Save to AlphaPose-style JSON file
    with open(output_path, "w") as f:
        json.dump(alphapose_format, f)
