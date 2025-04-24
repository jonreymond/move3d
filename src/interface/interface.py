# Before running, install required packages:
#   pip install gradio imageio
# Note: for testing without valid MP4 parsing, we stub out processing to avoid heavy video parsing delays.

import gradio as gr
import imageio
import os
import shutil
import tempfile
import numpy as np

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

    # Event: Process -> show frame and choices with loading spinner
    def on_process(video):
        img_path = process_until_decision(video)
        return (
            gr.update(value=img_path, visible=True),
            gr.update(visible=True),
            gr.update(visible=True)
        )
    process_btn.click(
        fn=on_process,
        inputs=[video_input],
        outputs=[img, choice, continue_btn],
        show_progress="full"
    )

    # Event: Continue -> hide inputs and button, show output video with spinner
    def on_continue(selected, video):
        output = continue_processing(video, selected)
        return (
            gr.update(visible=False),  # hide image
            gr.update(visible=False),  # hide choices
            gr.update(visible=False),  # hide continue button
            gr.update(value=output, visible=True)
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
