import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

matplotlib.use("Agg")


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


checkpoint = "../checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint)
video_dir = "notebooks/videos/bedroom"
seg_type = "box"
# scan all the JPEG frame names in this directory
frame_names = [
    p
    for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

ann_frame_idx = 0  # the frame index that we interact with
ann_obj_id = (
    1  # give a unique id to each object we interact with (it can be any integers)
)

if seg_type == "point":
    # Let's add a positive click at (x, y) = (210, 350) to get started
    points = np.array([[210, 350], [250, 220]], dtype=np.float32)
    # for labels, `1` means positive click and `0` means negative click
    labels = np.array([1, 1], np.int32)
    # 想加入新的click prompt的话就加points+labels
elif seg_type == "box":
    box = np.array([300, 0, 500, 400], dtype=np.float32)
elif seg_type == "multi":
    prompts = {}  # hold all the clicks we add for visualization
    # Let's add a 2nd negative click at (x, y) = (275, 175) to refine the first object
    # sending all clicks (and their labels) to `add_new_points_or_box`
    points1 = np.array([[200, 300], [275, 175]], dtype=np.float32)
    # for labels, `1` means positive click and `0` means negative click
    labels1 = np.array([1, 0], np.int32)
    ann_obj_id1 = 1
    prompts[ann_obj_id] = points1, labels1

    points2 = np.array([[400, 150]], dtype=np.float32)
    # for labels, `1` means positive click and `0` means negative click
    labels2 = np.array([1], np.int32)
    ann_obj_id2 = 3
    prompts[ann_obj_id2] = points2, labels2

else:
    raise NotImplementedError(f"{seg_type} not implemented yet.")

video_segments = {}  # video_segments contains the per-frame segmentation results

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    # video_dir only supports a video folder as input, filled with .jpg images.
    # demand: change to support video as input (decord reading)
    state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(state)
    # add new prompts and instantly get the output on the same frame
    if seg_type == "point":
        input_dicts = dict(
            inference_state=state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,  # point as prompt
            labels=labels,
        )
    elif seg_type == "box":
        input_dicts = dict(
            inference_state=state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            box=box,
        )
    elif seg_type == "multi":
        input1_dicts = dict(
            inference_state=state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id1,
            points=points1,
            labels=labels1,
        )

    if seg_type != "multi":
        frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            **input_dicts
        )
    else:
        frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            **input1_dicts
        )
        input2_dicts = dict(
            inference_state=state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id2,
            points=points2,
            labels=labels2,
        )
        frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            **input2_dicts
        )
    import pdb

    pdb.set_trace()
    vis_first_frame = True
    os.makedirs(f"notebooks/videos/bedroom_vis/{seg_type}", exist_ok=True)
    if vis_first_frame:
        # show the results on the current (interacted) frame
        plt.figure(figsize=(9, 6))
        plt.title(f"frame {ann_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
        if seg_type == "point":
            show_points(points, labels, plt.gca())
            show_mask(
                (out_mask_logits[0] > 0.0).cpu().numpy(),
                plt.gca(),
                obj_id=out_obj_ids[0],
            )

        elif seg_type == "box":
            show_box(box, plt.gca())
            show_mask(
                (out_mask_logits[0] > 0.0).cpu().numpy(),
                plt.gca(),
                obj_id=out_obj_ids[0],
            )

        elif seg_type == "multi":
            show_points(points2, labels2, plt.gca())
            for i, out_obj_id in enumerate(out_obj_ids):
                show_points(*prompts[out_obj_id], plt.gca())
                show_mask(
                    (out_mask_logits[i] > 0.0).cpu().numpy(),
                    plt.gca(),
                    obj_id=out_obj_id,
                )
        plt.savefig(f"notebooks/videos/bedroom_vis/{seg_type}/{ann_frame_idx}.png")
    # # propagate the prompts to get masklets throughout the video
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        state
    ):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    vis_video = True
    if vis_video:
        # render the segmentation results every few frames
        vis_frame_stride = 1
        plt.close("all")
        for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
            plt.figure(figsize=(6, 4))
            plt.title(f"frame {out_frame_idx}")
            plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
            plt.savefig(f"notebooks/videos/bedroom_vis/{seg_type}/{out_frame_idx}.png")
