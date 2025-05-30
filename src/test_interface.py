import tempfile, os
from pipeline import *


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

    output_video_path = os.path.join(config.output_folder, f'video_{filename}.mp4')
    output_json_path = os.path.join(config.output_json_folder, f'keypoints_{os.path.splitext(filename)[0]}.json')
    alpha_pose_output_path = os.path.join(config.output_json_folder, f'alpha_format_keypoints_{os.path.splitext(filename)[0]}.json')
    video_3d_path = os.path.join(config.video_3d_path, f'keypoints_{filename}')
    video = "D:/Downloads/trimmed_1s_C0057_middle.mp4"
    tmp = tempfile.gettempdir()
    # output_video_path = os.path.join(tmp, "processed.mp4")
    # output_json_path  = os.path.join(tmp, "processed.json")
    process_2D(model2d, video, output_video_path, output_json_path)

    detect_patient(video, output_json_path)


if __name__ == "__main__":
    main()
    print('done')