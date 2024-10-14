# Installation

Download the pretrained `SAM 2` checkpoints:

```bash
bash scripts/installation/download_checkpoints.sh
```

Download the pretrained `HOI Detector` checkpoints from [link](https://drive.google.com/file/d/1H2tWsZkS7tDF8q1-jdjx6V9XrK25EDbE/view)

### Installation

We use `python=3.10`, as well as `torch >= 2.3.1`, `torchvision>=0.18.1` and `cuda-12.1` in our environment.

```bash
conda create -n hoi_sam python=3.10
pip install cython
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
pip install -r requirements.txt
```
You can download the [CUDA Toolkit 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local), and set the environment variable manually as follows:

```bash
export CUDA_HOME=/path/to/cuda-12.1/
```

Install `Segment Anything 2`:

```bash
cd segment-anything-2
python setup.py build_ext --inplace
```

Install `HOI Detectir`:

```bash
cd hand_object_detector/lib
pip install -e .
```

Install detectron
```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

Install linters
```bash
python -m pip install pre-commit
pre-commit install
```