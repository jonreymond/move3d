# pip install rtmlib -i https://pypi.org/simple
# pip install onnxruntime

import cv2
from rtmlib import Body, draw_skeleton
import onnxruntime
import os

trial_video_path = os.path.join('..', '3dmotion', '20250326','camera', 'C0112.mp4')

def main():
    # Initialize the pose estimation model
    body = Body(
        mode='balanced',  # Options: 'performance', 'lightweight', 'balanced'
        backend='onnxruntime',
        device='cpu'
    )

    # Open the video file
    video_path = trial_video_path
    cap = cv2.VideoCapture(video_path)

    # Prepare to write the output video
    output_path = 'trial_C0112.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform pose estimation
        keypoints, scores = body(frame)

        # Draw the skeleton on the frame
        frame = draw_skeleton(frame, keypoints, scores, kpt_thr=0.5)

        # Write the frame to the output video
        out.write(frame)

    # Release resources
    cap.release()   
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()