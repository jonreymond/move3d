import kineticstoolkit.lab as ktk

c3d_contents = ktk.read_c3d("3dmotion/vicon/T2G002_20250326_PostRehab_GA3_rec03.c3d")

c3d_contents["Analogs"].plot(["Left Rectus femoris", "Left Semimembranosus"])



