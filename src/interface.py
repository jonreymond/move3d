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
from utils import *
from pathlib import Path


# Step 1: process until user decision point
# Save extracted frame into a temporary directory so Gradio can access it
# Here we stub by creating a placeholder image

def process_until_decision(model2d, video_path):
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

    output_f = Path(config.output_folder)


    output_video_path = os.path.join(config.output_folder, f'keypoints_{filename}.mp4')
    output_json_path = os.path.join(config.output_json_folder, f'keypoints_{os.path.splitext(filename)[0]}.json')
    alpha_pose_output_path = os.path.join(config.output_json_folder, f'alpha_format_keypoints_{os.path.splitext(filename)[0]}.json')
    video_3d_path = os.path.join(config.video_3d_path, f'keypoints_{filename}')
    
    if not os.path.exists(video_3d_path):
        os.makedirs(video_3d_path)
                
            
    # return the JSON data as a dict 
    # first_step = lambda video_path : process_2D(model2d, video_path, output_video_path, output_json_path)

    # Event: Process -> show frame and choices with loading spinner
    # def on_process(video):
    #     yield (
    #         gr.update(visible=False),   # img
    #         gr.update(visible=False),   # choice
    #         gr.update(visible=False),   # continue_btn
    #         gr.update(value="⏳ Processing…", visible=True)  # status
    #     )
    #     process_2D(model2d, video, output_video_path, output_json_path)
    #     nb_people, image = detect_patient_2(video, output_json_path)
    #     tmp_dir = tempfile.gettempdir()
    #     output_path = os.path.join(tmp_dir, "first_frame.png")
    #     imageio.imwrite(output_path, image)
    #     yield (
    #         gr.update(value=output_path, visible=True),  # img
    #         gr.update(visible=True),                      # choice
    #         gr.update(visible=True),                      # continue_btn
    #         gr.update(value="", visible=False)            # status
    #     )
    
    def on_process(video):
    # 1) Hide everything & show “Processing…” status
        yield (
            gr.update(visible=False),   # img
            gr.update(visible=False),   # nb_people_text
            gr.update(visible=False),   # choice
            gr.update(visible=False),   # continue_btn
            gr.update(value="⏳ Processing…", visible=True),  # status
            gr.update(visible=False),             # validation_result
            0  # reset people‐count state to zero  # validation_result
        )

        # 2) Run your heavy-lifting
        process_2D(model2d, video, output_video_path, output_json_path)
        nb_people, image = detect_patient_2(video, output_json_path)

        # 3) Save the first frame PNG
        tmp_dir = tempfile.gettempdir()
        output_path = os.path.join(tmp_dir, "first_frame.png")
        imageio.imwrite(output_path, image)

        # 4) Show the results & clear status
        yield (
            gr.update(value=output_path, visible=True),                         # img
            gr.update(value=f"Detected **{nb_people}** people", visible=True),   # nb_people_text
            gr.update(visible=True),                                            # choice
            gr.update(visible=True),                                            # continue_btn
            gr.update(value="", visible=False),                                  # status
            gr.update(visible=True),
            nb_people
        )
        # return(nb_people)

    def validate_choice(choice_input, nb_people):
        try:
            idx = int(choice_input)
            if 0 <= idx < nb_people:
                return f"✅ Valid choice: {idx}"
            else:
                return f"❌ Please enter an integer between 0 and {nb_people-1}."
        except:
            return "❌ Invalid input. Please enter a number."

    # Event: Continue -> hide inputs and button, show output video with spinner
    def on_continue(selected):
        # Hide everything & show “Processing…” status
        yield (
            gr.update(visible=False),  # img
            gr.update(visible=False),  # nb_people_text
            gr.update(visible=False),  # choice
            gr.update(visible=False),  # continue_btn
            gr.update(value="⏳ Processing…", visible=True),  # status
            gr.update(visible=False),  # validation_result
            gr.update(visible=True)  # video_out
        )
        idx = int(selected)
        raw_data = get_new_data(output_json_path, idx)
        convert_JSON_MB_format(raw_data, alpha_pose_output_path)
        motion_BERT(alpha_pose_output_path, selected, video_3d_path)
        yield (
            gr.update(visible=False),  # img
            gr.update(visible=False),  # nb_people_text
            gr.update(visible=False),  # choice
            gr.update(visible=False),  # continue_btn
            gr.update(value="", visible=False),  # status
            gr.update(visible=True),  # validation_result
            gr.update(visible=True)  # video_out
        )
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


                status = gr.Markdown("", visible=False)          # <-- new status area
                nb_people_text = gr.Markdown("", visible=False) 
                img = gr.Image(visible=False)
                choice = gr.Textbox(label="Input", lines=2, placeholder="Type your question here...", visible=False)
                continue_btn = gr.Button("Continue", visible=False)
                validation_result = gr.Textbox(label="Validation Result", interactive=False, visible=False)
                nb_people_state    = gr.State(value=0)

            with gr.Column():
                gr.Markdown("## Result")
                video_out = gr.Video(label="Result Video")

        
            # process_btn.click(
            #     fn=on_process,
            #     inputs=[video_input],
            #     outputs=[img, choice, continue_btn, status],
            # )

            process_btn.click(
                fn=on_process,
                inputs=[video_input],
                outputs=[
                    img,
                    nb_people_text,
                    choice,
                    continue_btn,
                    status,
                    validation_result,
                    nb_people_state
                ],
            )

            choice.change(
                fn=validate_choice,
                inputs=[choice, nb_people_state],
                outputs=[validation_result],
            )

            continue_btn.click(
                fn=on_continue,
                inputs=[choice],
                outputs=[
                    img,
                    nb_people_text,
                    choice,
                    continue_btn,
                    status,
                    validation_result,
                    video_out
                ],
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