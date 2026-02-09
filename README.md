# Social-LOVON: a Socially-Aware Extension of LOVON: Legged Open Vocabulary Object Navigator. 

## TODO:
- [ ] Trajectory Prediction
    - [ ] Coordinate Transformation
- [ ] Social Force Model
- [ ] Action Shield Logic
    - [ ] Implementation
- [ ] Simulation Environment

## Installation

```bash
# 1. Create a virtual environment
conda create -n lovon_env python=3.8 -y
# Activate the environment
conda activate lovon_env

# 2. Install PyTorch (Choose based on your GPU configuration)
# For CPU-only
pip install torch>=1.10.0
# For GPU, install manually according to your CUDA version (e.g., CUDA 11.7)
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# 3. Install other dependencies
pip install -r requirements.txt
```

## Usage

```bash
python deploy/deploy.py --simulation_mode
```


## Contributing

## License