from bs4 import BeautifulSoup
from datetime import datetime
import numpy as np
import cv2

import kineticstoolkit.geometry as geom
import kineticstoolkit.lab as ktk

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


"""
-----------------------------------------------
C3D LOADING AND TRANSFORMATIONS
-----------------------------------------------
"""

def load_c3d(path):
    """loading a c3d file

    Args:
        path (str): path to c3d file

    Returns:
        np.array: the resulted numpy array with shape (num frames, height, width, )
    """
    c3d_file = ktk.read_c3d(path)


    return c3d_file["Points"]

def transform_c3d_into_local_hip_ankle(c3d_points):
    """
    transform the c3d points into local hip and ankle coordinates
    Args:
        c3d_points (np.array): the c3d points with shape (num frames, num points, 3)

    Returns:
        np.array: the resulted numpy array with shape (num frames, num points, 3)
    """
    # 2) Extract raw (N×4) marker arrays
    hip4   = c3d_points.data["LHIP"]
    knee4  = c3d_points.data["LKNE"]
    ankle4 = c3d_points.data["LANK"]

    # 3) FEMUR CS: build 3D vectors (drop homogeneous col)
    O_f3   = hip4[:, :3]                          # origin
    Y_f3   = (hip4 - knee4)[:, :3]                # y axis (cranial)
    YZ_f3  = (hip4 - ankle4)[:, :3]               # in‐plane vector

    # normalize directions
    def norm(v): return v / np.linalg.norm(v, axis=1, keepdims=True)
    Y_f3, YZ_f3 = norm(Y_f3), norm(YZ_f3)

    # promote to homogeneous (N×4)
    N      = c3d_points.time.shape[0]
    O_f4   = np.hstack([O_f3,  np.ones((N,1))])   # w=1 for origin
    Y_f4   = np.hstack([Y_f3,  np.zeros((N,1))])  # w=0 for direction
    YZ_f4  = np.hstack([YZ_f3, np.zeros((N,1))])

    # create femur transforms via y & yz → returns (N,4,4) ndarray :contentReference[oaicite:1]{index=1}
    femur_tf = geom.create_transform_series(
        positions=O_f4,
        y=Y_f4,
        yz=YZ_f4,
        length=N
    )

    # 4) ANKLE CS **without** toe: use tibia axis + global vertical as plane vector
    O_a3   = ankle4[:, :3]                        # origin at ankle
    Y_a3   = norm((knee4 - ankle4)[:, :3])        # tibial long axis

    # global vertical direction vector, homogeneous
    Z0_4   = np.tile([0, 1, 0, 0], (N,1))         # w=0 for direction

    O_a4   = np.hstack([O_a3,  np.ones((N,1))])
    Y_a4   = np.hstack([Y_a3,  np.zeros((N,1))])

    # create ankle transforms via y & yz=global vertical
    ankle_tf = geom.create_transform_series(
        positions=O_a4,
        y=Y_a4,
        yz=Z0_4,
        length=N
    )

    # 5) Wrap each transform array in a ktk.TimeSeries
    fem_ts = ktk.TimeSeries(); fem_ts.time = c3d_points.time.copy()
    fem_ts.data["FemurCS"] = femur_tf

    ank_ts = ktk.TimeSeries(); ank_ts.time = c3d_points.time.copy()
    ank_ts.data["AnkleCS"] = ankle_tf  

    combined = c3d_points.merge(fem_ts).merge(ank_ts)

    hip_to_ankle = ktk.geometry.get_local_coordinates(
        combined.data["FemurCS"], combined.data["AnkleCS"]
        )

    
    return hip_to_ankle

def calculate_c3d_angles(local_c3d_points):
    """calculate the angles of the c3d points

    Args:
        c3d_points (np.array): the c3d points with shape (num frames, num points, 3)

    Returns:
        np.array: the resulted numpy array with shape (num frames, num angles)
    """
    angles = ktk.geometry.get_angles(local_c3d_points, "ZXY")

    return np.unwrap(angles)
"""
-----------------------------------------------
ANGLE CALCULATION FROM 3D POINTS
-----------------------------------------------
"""

