
import os
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')  # base style

# override background + details
plt.rcParams.update({
    "axes.facecolor": "#f9f9f9",     # light beige inside plot
    "figure.facecolor": "#f2f2f2",   # light gray canvas
    "grid.color": "white",           # white grid
    "axes.edgecolor": "#dddddd",
    "axes.labelcolor": "#333333",
    "xtick.color": "#666666",
    "ytick.color": "#666666",
    "legend.frameon": False,
    "font.size": 13,
    "font.family": "DejaVu Sans"
})
def coco17_to_h36m17(x):
    y = np.zeros((x.shape[0], 17, 3))  # input: (frames, 17, 3)
    y[:, 0] = (x[:, 11] + x[:, 12]) / 2  # Hip center
    y[:, 1] = x[:, 12]                  # RHip
    y[:, 2] = x[:, 14]                  # RKnee
    y[:, 3] = x[:, 16]                  # RFoot
    y[:, 4] = x[:, 11]                  # LHip
    y[:, 5] = x[:, 13]                  # LKnee
    y[:, 6] = x[:, 15]                  # LFoot
    y[:, 7] = (x[:, 5] + x[:, 6]) / 2   # Spine midpoint between shoulders
    y[:, 8] = x[:, 0]                   # Thorax ≈ Nose
    y[:, 9] = x[:, 0]                   # Neck
    y[:,10] = x[:, 0]                   # Head
    y[:,11] = x[:, 5]                   # LShoulder
    y[:,12] = x[:, 7]                   # LElbow
    y[:,13] = x[:, 9]                   # LWrist
    y[:,14] = x[:, 6]                   # RShoulder
    y[:,15] = x[:, 8]                   # RElbow
    y[:,16] = x[:,10]                   # RWrist
    return y

def compute_angle(a, b, c):
    """Returns angle at b formed by a-b-c (in degrees)."""
    ba = a - b
    bc = c - b
    cos_angle = np.sum(ba * bc, axis=-1) / (
        np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1)
    )
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)



def main():
    numpy_3d_path = os.path.join('data', '3d-reconstruction', 'keypoints_trimmed_1s_C0057_middle.mp4', 'X3D.npy')
    add_str = '1 sec'
    keypoints_3d = np.load(numpy_3d_path)
    
    # Convert to H36M format
    h36m = coco17_to_h36m17(keypoints_3d)
    
    # Indices
    RHIP, RKNE, RFOOT = 4, 5, 6
    LHIP, LKNE, LFOOT = 1, 2, 3
    # Compute joint angles
    left_knee_angle = compute_angle(h36m[:, LHIP], h36m[:, LKNE], h36m[:, LFOOT])
    right_knee_angle = compute_angle(h36m[:, RHIP], h36m[:, RKNE], h36m[:, RFOOT])
    knee_diff = np.abs(left_knee_angle - right_knee_angle)

    # Hip angle 
    left_hip_angle = compute_angle(h36m[:, 7], h36m[:, 1], h36m[:, 2])  # Spine-RHip-RKnee
    right_hip_angle = compute_angle(h36m[:, 7], h36m[:, 4], h36m[:, 5])   # Spine-LHip-LKnee
    
    # Range of Motion (ROM)
    left_rom = np.max(left_knee_angle) - np.min(left_knee_angle)
    right_rom = np.max(right_knee_angle) - np.min(right_knee_angle)
    red = [250/255, 82/255, 91/255]
    blue = [25/255,211/255,197/255]
    violet = [77/255,38/255,122/255]
    
    # Plot: Time-Series of Left vs Right Knee Angle
    plt.figure(figsize=(10, 4))
    plt.plot(left_knee_angle, label="Left Knee", color=blue)
    plt.plot(right_knee_angle, label="Right Knee", color=red)
    plt.title("Knee Flexion Angles Over Time")
    plt.xlabel("Frame")
    plt.ylabel("Angle (degrees)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'Knee Flexion Angles Over Time {add_str}')
    plt.show()

    # Plot: Time-Series of Left vs Right Hig Angle
    plt.figure(figsize=(10, 4))
    plt.plot(left_hip_angle, label="Left Hip", color=blue)
    plt.plot(right_hip_angle, label="Right Hip", color=red)
    plt.title("Hip Flexion Angles Over Time")
    plt.xlabel("Frame")
    plt.ylabel("Angle (degrees)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Hip Flexion Angles Over Time {add_str}")
    plt.show()

    # Plot: Knee Angle Symmetry (Difference)
    plt.figure(figsize=(10, 3))
    plt.plot(knee_diff, label="|L - R| Knee Angle", color=violet)
    plt.title("Knee Angle Asymmetry")
    plt.xlabel("Frame")
    plt.ylabel("Angle Difference (°)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Knee Angle Asymmetry {add_str}")
    plt.show()

    # Plot: Range of Motion Comparison
    plt.figure(figsize=(5, 4))
    plt.bar(["Left Knee", "Right Knee"], [left_rom, right_rom], color=[blue, red])
    plt.title("Knee Range of Motion")
    plt.ylabel("ROM (degrees)")
    plt.tight_layout()
    plt.savefig(f"Knee Range of Motion {add_str}")
    plt.show()

if __name__ == "__main__":
    main()
    print('done')