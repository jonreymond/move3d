# move3d

## Installation
Make sure to have installed Conda or miniconda in your machine. Then run `conda env create -f environment.yml`.
You could also use mamba to build the environment quicker. First, run `conda install -n base -c conda-forge mamba` if mamba is not installed, then `mamba env create -f environment.yml`.

To install the environment, run the following pipelines: 
`curl -LsSf https://astral.sh/uv/install.sh | sh`
`uv sync`
`uv run src/estimation_2d.py`

