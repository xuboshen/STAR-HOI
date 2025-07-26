# Segment and Track Any Hand-Object Interactions (Segment Any HOI)

HOI-SAM: Segment and Track Hand-Object Interaction in offline Videos with [HOI Detector](https://openaccess.thecvf.com/content_CVPR_2020/papers/Shan_Understanding_Human_Hands_in_Contact_at_Internet_Scale_CVPR_2020_paper.pdf) and [SAM 2](https://arxiv.org/abs/2408.00714).

**🔥 Project Highlight**

In this repo, we've supported the following codes with **simple implementations**:
- **Image-Level Detection and Segmentation**: Firstly detect HOIs, then segment HOIs with the HOI prompts in image.
- **Video-level Tracking and Segmentation**: Given prompts of the initial frame, track all HOIs in the videos;
- **Evaluation of HOI Segmentation on VISOR(EK-100)**: only on eval set.


## Installation

See [INSTALL.md](docs/INSTALL.md)

### Thanks:

[detectron2](https://github.com/facebookresearch/detectron2)
[visor-hos](https://github.com/epic-kitchens/VISOR-HOS)
[sam2](https://github.com/facebookresearch/segment-anything-2)
