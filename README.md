# Segment and Track Any Hand-Object Interactions (HOI-SAM)

HOI-SAM: Segment and Track Hand-Object Interaction in offline Videos with [HOI Detector](https://openaccess.thecvf.com/content_CVPR_2020/papers/Shan_Understanding_Human_Hands_in_Contact_at_Internet_Scale_CVPR_2020_paper.pdf) and [SAM 2](https://arxiv.org/abs/2408.00714).

**🔥 Project Highlight**

In this repo, we've supported the following codes with **simple implementations**:
- **Image-Level Detection and Segmentation**: Firstly detect HOIs, then segment HOIs with the HOI prompts in image.
- **Video-level Tracking and Segmentation**: Given prompts of the initial frame, track all HOIs in the videos;
- **Evaluation of HOI Segmentation on VISOR(EK-100)**: only on eval set.


## News

- `2024/08/20`: Support **Florence-2 SAM 2 Image Demo** which includes `dense region caption`, `object detection`, `phrase grounding`, and cascaded auto-label pipeline `caption + phrase grounding`.
- `2024/08/09`: Support **Ground and Track New Object** throughout the whole videos. This feature is still under development now. Credits to [Shuo Shen](https://github.com/ShuoShenDe).
- `2024/08/07`: Support **Custom Video Inputs**, users need only submit their video file (e.g. `.mp4` file) with specific text prompts to get an impressive demo videos.

## Contents
- [Installation](#installation)
- [Grounded SAM 2 Demos](#grounded-sam-2-demos)
  - [Grounded SAM 2 Image Demo](#grounded-sam-2-image-demo-with-grounding-dino)
  - [Grounded SAM 2 Image Demo (with Grounding DINO 1.5 & 1.6)](#grounded-sam-2-image-demo-with-grounding-dino-15--16)
  - [Grounded SAM 2 Video Object Tracking Demo](#grounded-sam-2-video-object-tracking-demo)
  - [Grounded SAM 2 Video Object Tracking Demo (with Grounding DINO 1.5 & 1.6)](#grounded-sam-2-video-object-tracking-demo-with-grounding-dino-15--16)
  - [Grounded SAM 2 Video Object Tracking with Custom Video Input (using Grounding DINO)](#grounded-sam-2-video-object-tracking-demo-with-custom-video-input-with-grounding-dino)
  - [Grounded SAM 2 Video Object Tracking with Custom Video Input (using Grounding DINO 1.5 & 1.6)](#grounded-sam-2-video-object-tracking-demo-with-custom-video-input-with-grounding-dino-15--16)
  - [Grounded SAM 2 Video Object Tracking with Continues ID (using Grounding DINO)](#grounded-sam-2-video-object-tracking-with-continuous-id-with-grounding-dino)
- [Grounded SAM 2 Florence-2 Demos](#grounded-sam-2-florence-2-demos)
  - [Grounded SAM 2 Florence-2 Image Demo](#grounded-sam-2-florence-2-image-demo-updating)
  - [Grounded SAM 2 Florence-2 Image Auto-Labeling Demo](#grounded-sam-2-florence-2-image-auto-labeling-demo)
- [Citation](#citation)



## Installation

Download the pretrained `SAM 2` checkpoints:

```bash
bash scripts/installation/download_checkpoints.sh
```

Download the pretrained `HOI Detector` checkpoints from [link](https://drive.google.com/file/d/1H2tWsZkS7tDF8q1-jdjx6V9XrK25EDbE/view)

### Installation

We use `python=3.10`, as well as `torch >= 2.3.1`, `torchvision>=0.18.1` and `cuda-12.1` in our environment.

```bash
conda create -n hoi_sam python=3.10
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
```
You can download the [CUDA Toolkit 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local), and set the environment variable manually as follows:

```bash
export CUDA_HOME=/path/to/cuda-12.1/
```

Install `Segment Anything 2`:

```bash
cd segment-anything-2
pip install -e .
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


\
\
\
\
\
\
\
\
\


### Installation with docker
Build the Docker image and Run the Docker container:

```
cd Grounded-SAM-2
make build-image
make run
```
After executing these commands, you will be inside the Docker environment. The working directory within the container is set to: `/home/appuser/Grounded-SAM-2`

Once inside the Docker environment, you can start the demo by running:
```
python grounded_sam2_tracking_demo.py
```

## Grounded SAM 2 Demos
### Grounded SAM 2 Image Demo (with Grounding DINO)
Note that `Grounding DINO` has already been supported in [Huggingface](https://huggingface.co/IDEA-Research/grounding-dino-tiny), so we provide two choices for running `Grounded SAM 2` model:
- Use huggingface API to inference Grounding DINO (which is simple and clear)

```bash
python grounded_sam2_hf_model_demo.py
```

> [!NOTE]
> 🚨 If you encounter network issues while using the `HuggingFace` model, you can resolve them by setting the appropriate mirror source as `export HF_ENDPOINT=https://hf-mirror.com`

- Load local pretrained Grounding DINO checkpoint and inference with Grounding DINO original API (make sure you've already downloaded the pretrained checkpoint)

```bash
python grounded_sam2_local_demo.py
```


### Grounded SAM 2 Image Demo (with Grounding DINO 1.5 & 1.6)

We've already released our most capable open-set detection model [Grounding DINO 1.5 & 1.6](https://github.com/IDEA-Research/Grounding-DINO-1.5-API), which can be combined with SAM 2 for stronger open-set detection and segmentation capability. You can apply the API token first and run Grounded SAM 2 with Grounding DINO 1.5 as follows:

Install the latest DDS cloudapi:

```bash
pip install dds-cloudapi-sdk
```

Apply your API token from our official website here: [request API token](https://deepdataspace.com/request_api).

```bash
python grounded_sam2_gd1.5_demo.py
```

### Grounded SAM 2 Video Object Tracking Demo

Based on the strong tracking capability of SAM 2, we can combined it with Grounding DINO for open-set object segmentation and tracking. You can run the following scripts to get the tracking results with Grounded SAM 2:

```bash
python grounded_sam2_tracking_demo.py
```

- The tracking results of each frame will be saved in `./tracking_results`
- The video will be save as `children_tracking_demo_video.mp4`
- You can refine this file with different text prompt and video clips yourself to get more tracking results.
- We only prompt the first video frame with Grounding DINO here for simple usage.

#### Support Various Prompt Type for Tracking

We've supported different types of prompt for Grounded SAM 2 tracking demo:

- **Point Prompt**: In order to **get a stable segmentation results**, we re-use the SAM 2 image predictor to get the prediction mask from each object based on Grounding DINO box outputs, then we **uniformly sample points from the prediction mask** as point prompts for SAM 2 video predictor
- **Box Prompt**: We directly use the box outputs from Grounding DINO as box prompts for SAM 2 video predictor
- **Mask Prompt**: We use the SAM 2 mask prediction results based on Grounding DINO box outputs as mask prompt for SAM 2 video predictor.

![Grounded SAM 2 Tracking Pipeline](./assets/g_sam2_tracking_pipeline_vis_new.png)


### Grounded SAM 2 Video Object Tracking Demo (with Grounding DINO 1.5 & 1.6)

We've also support video object tracking demo based on our stronger `Grounding DINO 1.5` model and `SAM 2`, you can try the following demo after applying the API keys for running `Grounding DINO 1.5`:

```bash
python grounded_sam2_tracking_demo_with_gd1.5.py
```

### Grounded SAM 2 Video Object Tracking Demo with Custom Video Input (with Grounding DINO)

Users can upload their own video file (e.g. `assets/hippopotamus.mp4`) and specify their custom text prompts for grounding and tracking with Grounding DINO and SAM 2 by using the following scripts:

```bash
python grounded_sam2_tracking_demo_custom_video_input_gd1.0_hf_model.py
```

### Grounded SAM 2 Video Object Tracking Demo with Custom Video Input (with Grounding DINO 1.5 & 1.6)

Users can upload their own video file (e.g. `assets/hippopotamus.mp4`) and specify their custom text prompts for grounding and tracking with Grounding DINO 1.5 and SAM 2 by using the following scripts:

```bash
python grounded_sam2_tracking_demo_custom_video_input_gd1.5.py
```

You can specify the params in this file:

```python
VIDEO_PATH = "./assets/hippopotamus.mp4"
TEXT_PROMPT = "hippopotamus."
OUTPUT_VIDEO_PATH = "./hippopotamus_tracking_demo.mp4"
API_TOKEN_FOR_GD1_5 = "Your API token" # api token for G-DINO 1.5
PROMPT_TYPE_FOR_VIDEO = "mask" # using SAM 2 mask prediction as prompt for video predictor
```

After running our demo code, you can get the tracking results as follows:

[![Video Name](./assets/hippopotamus_seg.jpg)](https://github.com/user-attachments/assets/1fbdc6f4-3e50-4221-9600-98c397beecdf)

And we will automatically save the tracking visualization results in `OUTPUT_VIDEO_PATH`.

> [!WARNING]
> We initialize the box prompts on the first frame of the input video. If you want to start from different frame, you can refine `ann_frame_idx` by yourself in our code.

### Grounded-SAM-2 Video Object Tracking with Continuous ID (with Grounding DINO)

In above demos, we only prompt Grounded SAM 2 in specific frame, which may not be friendly to find new object during the whole video. In this demo, we try to **find new objects** and assign them with new ID across the whole video, this function is **still under develop**. it's not that stable now.

Users can upload their own video files and specify custom text prompts for grounding and tracking using the Grounding DINO and SAM 2 frameworks. To do this, execute the script:


```bash 
python grounded_sam2_tracking_demo_with_continuous_id.py
```

You can customize various parameters including:

- `text`: The grounding text prompt.
- `video_dir`: Directory containing the video files.
- `output_dir`: Directory to save the processed output.
- `output_video_path`: Path for the output video.
- `step`: Frame stepping for processing.
- `box_threshold`: box threshold for groundingdino model
- `text_threshold`: text threshold for groundingdino model
Note: This method supports only the mask type of text prompt.

After running our demo code, you can get the tracking results as follows:

[![Video Name](./assets/tracking_car_mask_1.jpg)](https://github.com/user-attachments/assets/d3f91ad0-3d32-43c4-a0dc-0bed661415f4)

If you want to try `Grounding DINO 1.5` model, you can run the following scripts after setting your API token:

```bash
python grounded_sam2_tracking_demo_with_continuous_id_gd1.5.py
```

### Grounded-SAM-2 Video Object Tracking with Continuous ID plus Reverse Tracking(with Grounding DINO)
This method could simply cover the whole lifetime of the object
```bash
python grounded_sam2_tracking_demo_with_continuous_id_plus.py

```

### Citation

### Thanks:

[detectron2](https://github.com/facebookresearch/detectron2)
[visor-hos](https://github.com/epic-kitchens/VISOR-HOS)
[sam2](https://github.com/facebookresearch/segment-anything-2)