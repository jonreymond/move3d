from bs4 import BeautifulSoup
from datetime import datetime
import numpy as np
import cv2

from typing import Optional, Union

def read_timestamp(file_path: str) -> Optional[Union[datetime, str]]:
    """
    Extract a timestamp from either:
     - a ProfessionalDisc NonRealTimeMeta XML (CreationDate/@value or root@lastUpdate),
     - or a Qualisys Trial .history XML (Param name="Capture Date" @value).
    
    Returns:
      - datetime.datetime on successful parse,
      - the raw string if parsing fails,
      - or None if no timestamp is found.
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        xml = f.read()
    
    soup = BeautifulSoup(xml, "xml")
    
    cd = soup.find('CreationDate')
    if cd and cd.has_attr('value'):
        val = cd['value']
        try:
            return datetime.fromisoformat(val)
        except ValueError:
            return val
    
    root = soup.find()
    if root and root.has_attr('lastUpdate'):
        val = root['lastUpdate']
        try:
            return datetime.fromisoformat(val)
        except ValueError:
            return val
    
    param = soup.find('Param', attrs={'name': 'Capture Date'})
    if param and param.has_attr('value'):
        val = param['value']  # e.g. "2025-Mar-26 16:38:04"
        try:
            return datetime.strptime(val, "%Y-%b-%d %H:%M:%S")
        except ValueError:
            return val
    
    return None
    
 
    
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