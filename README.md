# Mamba-based Multi-Object Tracking Scripts

This repository contains multi-object tracking (MOT) scripts using Mamba-based vision models for construction worker tracking.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Preparation](#model-preparation)
- [Inference Commands](#inference-commands)
- [Output Format](#output-format)

## Overview

### Pipeline

1. **Detection**: VMamba, MambaVision, or GroundingDINO detects objects/construction workers
2. **ReID Feature Extraction**: ResNet-50 or Vision Mamba (Vim) extracts visual appearance features from each detected bounding box
3. **Tracking**: A tracker associates boxes across frames using IoU, center distance, and ReID feature similarity
4. **Output**: Results are written in MOT format

### Supported Scripts

| Script | Detector | ReID Model | Description |
|--------|----------|------------|-------------|
| `vmamba_mot.py` | VMamba (Mask R-CNN) | ResNet-50 | VMamba-based detection with ResNet ReID |
| `mambavision_mot.py` | MambaVision | - | MambaVision detection with built-in tracking |
| `gdino_vim_mot.py` | GroundingDINO | Vim (Vision Mamba) | GroundingDINO detection with Vim ReID |

## Installation

### Requirements

- Python 3.10.13
- PyTorch 2.1.1+CUDA 12.1+ (recommended)
```bash
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121

```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install Mamba Dependencies

```bash
#test cxx11abi flag
python -c "import torch;print(torch._C._GLIBCXX_USE_CXX11_ABI)"

#if  False

#download  causal-conv1d  and upload it to server
# https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/causal_conv1d-1.5.0.post8+cu12torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

#download  mamba_ssm  and upload it  to server
# https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

pip install causal_conv1d-1.5.0.post8+cu12torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

pip install mamba_ssm-2.2.4+cu12torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

#if  True

#download  causal-conv1d  and upload it to server
# https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/causal_conv1d-1.5.0.post8+cu12torch2.1cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

#download  mamba_ssm  and upload it  to server
# https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.1cxx11abiTRUE-cp310-cp310-linux_x86_64.whl


pip install  ...

```


### Install MambaVision Dependencies


```bash
pip install mambavision

```


### Verify Installation

```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "import mmdet; print(mmdet.__version__)"
python -c "from transformers import AutoModelForZeroShotObjectDetection"
```

## Data Preparation

### Dataset Structure

The scripts expect the following directory structure:

```
data/
└── dwtest/
    ├── images/
    │   └── val/
    │       ├── 08_51_23-09_21_19-1796730/
    │       │   ├── 000001.jpg
    │       │   ├── 000002.jpg
    │       │   └── ...
    │       └── 08_51_59-08_57_36-337793/
    └── gt/
        └── val/
            ├── 08_51_23-09_21_19-1796730/
            │   └── gt.txt
            └── 08_51_59-08_57_36-337793/
                └── gt.txt
```

### Image Requirements

- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`
- Frame naming: Numeric filenames (e.g., `000001.jpg`) are recommended for correct ordering
- Images should be organized by sequence/scene directories

### Ground Truth Format (Optional, for evaluation)

The MOT ground truth format:
```
frame_id,track_id,x1,y1,w,h,1,-1,-1,-1
```

## Model Preparation

### VMamba Models

```bash
# Download VMamba Mask R-CNN checkpoint
mkdir -p VMamba-main/checkpoint
wget -P VMamba-main/checkpoint https://huggingface.co/zhuyue/mamba-based-mask-rcnn/resolve/main/mask_rcnn_vssm_fpn_coco_base_epoch_11.pth
```

**Configuration files:**
- `mask_rcnn_vssm_fpn_coco_base.py` - VMamba base configuration

### MambaVision Models

```bash
# Download MambaVision checkpoint
mkdir -p MambaVision-main/ckp
wget -P MambaVision-main/ckp https://huggingface.co/NVlabs/MambaVision/resolve/main/cascade_mask_rcnn_mamba_vision_tiny_3x_coco.pth
```

**Configuration files:**
- `mamba_vision_tiny.py` - MambaVision tiny configuration

### Vim Models

```bash
# Download Vim checkpoint
mkdir -p Vim-main/ckp
wget -P Vim-main/ckp https://huggingface.co/hustvl/Vim/resolve/main/vim_b_midclstok_81p9acc.pth
```

### GroundingDINO Models

```bash
# Download GroundingDINO tiny model
mkdir -p IDEA-Research/grounding-dino-tiny
# Download from HuggingFace or ModelScope
```

## Inference Commands

### Basic Usage

```bash
# Using VMamba Detector + ResNet ReID
python vmamba_mot.py \
    --image_dir /path/to/images \
    --vmamba_config /path/to/mask_rcnn_vssm_fpn_coco_base.py \
    --vmamba_checkpoint /path/to/mask_rcnn_vssm_fpn_coco_base_epoch_11.pth \
    --output_dir ./outputs/vmamba_mot

# Using MambaVision Detector
python mambavision_mot.py \
    --image_dir /path/to/images \
    --mambavision_config /path/to/mamba_vision_tiny.py \
    --mambavision_checkpoint /path/to/cascade_mask_rcnn_mamba_vision_tiny_3x_coco.pth \
    --output_dir ./outputs/mambavision_mot

# Using GroundingDINO + Vim ReID
python gdino_vim_mot.py \
    --image_dir /path/to/images \
    --gdino_model /path/to/grounding-dino-tiny \
    --vim_model /path/to/vim_b_midclstok_81p9acc.pth \
    --output_dir ./outputs/vim_mot
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--image_dir` | varies | Input image directory |
| `--gt_root` | varies | Ground truth root (optional) |
| `--output_dir` | varies | Output directory for results |
| `--vmamba_config` | varies | VMamba config path |
| `--vmamba_checkpoint` | varies | VMamba checkpoint path |
| `--mambavision_config` | varies | MambaVision config path |
| `--mambavision_checkpoint` | varies | MambaVision checkpoint path |
| `--gdino_model` | varies | GroundingDINO model path |
| `--vim_model` | varies | Vim model checkpoint |
| `--prompt` | "person . construction worker . worker ." | Detection prompt (GroundingDINO) |
| `--batch_size` | 4 | ReID batch size |
| `--crop_pad` | 0.0 | Box expansion factor |
| `--box_threshold` | 0.35 | Detection box threshold |
| `--text_threshold` | 0.25 | Detection text threshold |
| `--vis` | False | Visualize results |
| `--show_stats` | False | Show performance statistics |

### Example with Custom Parameters

```bash
# VMamba tracking with visualization
python vmamba_mot.py \
    --image_dir /data/dwtest/images/train/seq001 \
    --output_dir ./outputs/vmamba_exp001 \
    --vmamba_config /path/to/mask_rcnn_vssm_fpn_coco_base.py \
    --vmamba_checkpoint /path/to/mask_rcnn_vssm_fpn_coco_base_epoch_11.pth \
    --box_threshold 0.4 \
    --vis \
    --show_stats

# GroundingDINO + Vim tracking
python gdino_vim_mot.py \
    --image_dir /data/dwtest/images/train/seq001 \
    --output_dir ./outputs/vim_exp001 \
    --gdino_model ./models/grounding-dino-tiny \
    --vim_model ./models/vim_b_midclstok_81p9acc.pth \
    --batch_size 8 \
    --crop_pad 0.1 \
    --vis
```

## Output Format

### MOT Format

Results are saved in standard MOT challenge format:
```
frame_id,track_id,x1,y1,w,h,score,1,-1
```

Example:
```
1,1,100.00,200.00,150.00,300.00,0.9500,1,-1
1,2,400.00,150.00,180.00,320.00,0.9200,1,-1
2,1,102.00,205.00,148.00,298.00,0.9400,1,-1
```

### Output Directory Structure

```
outputs/
└── vmamba_mot/
    └── seq001/
        ├── results.txt       # MOT format results
        └── vis/              # Visualized frames (if --vis)
            ├── 000001.jpg
            ├── 000002.jpg
            └── ...
```

## Performance Monitoring

Scripts support performance monitoring with `--show_stats`:
- GPU memory usage peak tracking
- Frames per second (FPS) statistics
- Model parameter statistics

## Troubleshooting

### Common Issues

1. **MMDetection import errors**: Ensure MMDet 3.x is installed and all modules are registered
2. **VMamba backbone not found**: Make sure VMamba classification is installed and `Backbone_VSSM` is registered
3. **CUDA out of memory**: Reduce `--batch_size` or use smaller model variants

## License

This project is for research purposes. Please follow the respective model licenses:
- VMamba: Apache 2.0
- MambaVision: NVIDIA License
- Vim: Apache 2.0
- GroundingDINO: Apache 2.0

## Acknowledgments

- [VMamba](https://github.com/VMamba/VMamba)
- [MambaVision](https://github.com/NVlabs/MambaVision)
- [Vim (Vision Mamba)](https://github.com/hustvl/Vim)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
