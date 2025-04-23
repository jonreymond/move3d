from bs4 import BeautifulSoup
from datetime import datetime
import numpy as np
import cv2

def read_time(file_path):
    with open(file_path, 'r') as f:
        data = f.read()

    # Passing the stored data inside
    Bs_data = BeautifulSoup(data, "xml")
    b_name = Bs_data.find('CreationDate')
    value = b_name['value']
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        # fallback: return the raw string
        return value
    
 
    
def load_mp4(path, resize_shape=None, skip_frames=1, max_frames=None):
    """loading a mp4 file

    Args:
        path (str): path to mp4 file
        resize_shape (tuple, optional): resize image with new shape. Defaults to None.
        skip_frames (int, optional): if we want to skip every X frames. Defaults to 1.
        max_frames (int, optional): the maximum frames count we want to load. Defaults to None.

    Returns:
        np.array: the resulted numpy array with shape (num frames, height, width, )
    """
    cap = cv2.VideoCapture(path)
    frames = []
    frame_count = 0

    while cap.isOpened():
        if frame_count % skip_frames == 0:
            
            ret, frame = cap.read()
            
            if resize_shape:
                frame = cv2.resize(frame, resize_shape)
                
            if not ret:
                break
            
            frames.append(frame)
            frame_count += 1
            
            if max_frames and frame_count >= max_frames:
                break

    cap.release()
    return np.array(frames)