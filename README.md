# move3d
Welcome in the repository of the project Move3D, for 3D motion reconstruction of Spinal Cord injury rehabilitation sessions. 

The pipeline takes as an input a MP4 video and output the reconstruction in 3D. The code will ask you to chose the person of interest, based on the first frame. It allows you to chose the person of interest for modeling. 

## Choose the videos
Put all the MP4 videos (in .mp4 or .MP4 format in the videos folder in the repository). The pipeline will read all of them and output the 3D reconstruction MP4 and the keypoints inside **reconstruction in data/3d-reconstruction/patient-folder/**.


## Conda environment installation
Install this conda environment if you have planned to run the code locally on your machine. Make sure to have installed Conda or miniconda in your machine. Then run `conda env create -f environment.yml`.
You could also use mamba to build the environment quicker. First, run `conda install -n base -c conda-forge mamba` if mamba is not installed, then `mamba env create -f environment.yml`.


## UV environment installation + execution
If you want to run the project in a GPU, you will need to install a specific UV environment. To install it, run the following command: 
- `curl -LsSf https://astral.sh/uv/install.sh | sh`

Once it is done, copy paste the command printed by the previous command looking like : `source $HOME/.local/bin/env`

Then continue with:
- `uv sync`
- `uv run src/estimation_2d.py`


