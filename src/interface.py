# Before running, install required packages:
#   pip install gradio imageio
# Note: for testing without valid MP4 parsing, we stub out processing to avoid heavy video parsing delays.
import hydra
from omegaconf import DictConfig
import gradio as gr
import imageio
import os
import shutil
import tempfile
import numpy as np
from estimation_2d import *

# Step 1: process until user decision point
# Save extracted frame into a temporary directory so Gradio can access it
# Here we stub by creating a placeholder image

def process_until_decision(video_path):
    tmp_dir = tempfile.gettempdir()
    image_path = os.path.join(tmp_dir, "frame0.png")
    # Stub: write a blank image for testing
    imageio.imwrite(image_path, 255 * np.ones((200, 200, 3), dtype=np.uint8))
    return image_path

# Step 2: continue processing based on user choice
# Copy the uploaded video to demonstrate output without parsing it

def continue_processing(video_path, choice):
    tmp_dir = tempfile.gettempdir()
    output_path = os.path.join(tmp_dir, "output.mp4")
    shutil.copyfile(video_path, output_path)
    return output_path



# Event: Continue -> hide inputs and button, show output video with spinner
def on_continue(selected, video):
    output = continue_processing(video, selected)
    return (
        gr.update(visible=False),  # hide image
        gr.update(visible=False),  # hide choices
        gr.update(visible=False),  # hide continue button
        gr.update(value=output, visible=True)
    )
    
    
# Event: Process -> show frame and choices with loading spinner
def on_process(video):
    img_path = process_until_decision(video)
    return (
        gr.update(value=img_path, visible=True),
        gr.update(visible=True),
        gr.update(visible=True)
    )
    

@hydra.main(version_base=None, 
            # config_path="../configs", 
            config_path=os.path.join(os.path.dirname(__file__), "../configs"),
            config_name="estimation_2d")
def main(config:DictConfig):
    
    model2d = Body(
        mode=config.model.mode,
        backend=config.model.backend,
        device=config.model.device)
    
    
    filename = "results"

    output_video_path = os.path.join(config.output_folder, f'keypoints_{filename}')
    output_json_path = os.path.join(config.output_json_folder, f'keypoints_{os.path.splitext(filename)[0]}.json')
    alpha_pose_output_path = os.path.join(config.output_json_folder, f'alpha_format_keypoints_{os.path.splitext(filename)[0]}.json')
    video_3d_path = os.path.join(config.video_3d_path, f'keypoints_{filename}')
    
    if not os.path.exists(video_3d_path):
        os.makedirs(video_3d_path)
                
            
    # return the JSON data as a dict 
    first_step = lambda video_path : process_2D(model2d, video_path, output_video_path, output_json_path)
    
    # Build Gradio interface with title and two-column layout
    with gr.Blocks(title="Video Review & Edit") as demo:
        # Visible header
        gr.Markdown("# Video Review & Edit")

        # Two-column layout
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Upload & Review")
                video_input = gr.Video(label="Upload MP4")
                process_btn = gr.Button("Process Video")

                img = gr.Image(visible=False)
                choice = gr.Radio(choices=[str(i) for i in range(5)], label="Pick a number", visible=False)
                continue_btn = gr.Button("Continue", visible=False)

            with gr.Column():
                gr.Markdown("## Result")
                video_out = gr.Video(label="Result Video")

        
        process_btn.click(
            fn=on_process,
            inputs=[video_input],
            outputs=[img, choice, continue_btn],
            show_progress="full"
        )

        
        continue_btn.click(
            fn=on_continue,
            inputs=[choice, video_input],
            outputs=[img, choice, continue_btn, video_out],
            show_progress="full"
        )

        # Launch on specific host and port without sharing publicly
        demo.launch(
            server_name="127.0.0.1", 
            server_port=7860, 
            share=False,
            allowed_paths=[tempfile.gettempdir()]
        )


if __name__ == "__main__":
    main()
    print('done')