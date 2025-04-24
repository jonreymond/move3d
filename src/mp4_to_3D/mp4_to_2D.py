import cv2
import os
import json
from rtmlib import Body, draw_skeleton

def get_current_wd():
    print(f"Current working directory: {os.getcwd()}")

def get_2D_repr_from_mp4(output_folder, output_json_folder,dev='cpu',video_folder='/pvc/scratch/3dmotion/20250326/camera'):
    
    body = Body(
        mode='balanced',
        backend='onnxruntime',
        device=dev,
    )

    # Loop through videos
    for filename in os.listdir(video_folder):
        if filename.endswith('.MP4'):
            video_path = os.path.join(video_folder, filename)
            output_video_path = os.path.join(output_folder, f'keypoints_{filename}')
            output_json_path = os.path.join(output_json_folder, f'keypoints_{os.path.splitext(filename)[0]}.json')

            print(f"Processing: {filename}")

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
                print(f"Skipping {filename} due to invalid FPS or frame size.")
                cap.release()
                continue
            
            out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

            all_keypoints = []
            all_scores = []

            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                keypoints, scores = body(frame)
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

            # Save keypoints to JSON
            json_data = {
                "video": filename,
                "keypoints": all_keypoints,
                "scores": all_scores
            }

            with open(output_json_path, 'w') as f:
                json.dump(json_data, f)

            print(f"Saved keypoints JSON: {output_json_path}")

    cv2.destroyAllWindows()
