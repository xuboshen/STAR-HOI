import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

matplotlib.use("Agg")


def show_mask(mask, ax, random_color=False, obj_id=None, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [
            cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours
        ]
        mask_image = cv2.drawContours(
            mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2
        )
    ax.imshow(mask_image)
    # return mask_image


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


def show_masks(
    image,
    masks,
    scores,
    point_coords=None,
    box_coords=None,
    input_labels=None,
    borders=True,
    thresh_sam_score=0.5,
    save_path="default",
):
    if box_coords is None:
        box_coords = np.array([0, 0, 0, 0])
    if box_coords.shape[0] == 1:
        box_coords = box_coords.reshape(-1)

    if len(box_coords.shape) == 1:

        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            show_mask(mask, plt.gca(), borders=borders)
            # plt.imshow(mask_image)
            if point_coords is not None:
                assert input_labels is not None
                show_points(point_coords, input_labels, plt.gca())
            if box_coords is not None:
                # boxes
                show_box(box_coords, plt.gca())
            if len(scores) > 1:
                plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis("off")
            plt.savefig(save_path, dpi=200)
            plt.close()
    elif len(box_coords.shape) == 2:
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        if len(masks.shape) == 4:  # N_prompt, Number_of_images, H, W
            masks = np.transpose(masks, (1, 0, 2, 3))
            scores = np.transpose(scores, (1, 0))
        for i, (mask, score) in enumerate(zip(masks, scores)):
            for msk, box, sco in zip(mask, box_coords, score):
                if sco < thresh_sam_score:
                    continue
                show_mask(msk, plt.gca(), random_color=True)
                show_box(box, plt.gca())
            # for i, (mask, score) in enumerate(zip(masks, scores)):
            #     print(i)
            #     plt.figure(figsize=(10, 10))
            #     plt.imshow(image)
            #     mask_image = show_mask(mask, plt.gca(), borders=borders)
            #     plt.imshow(mask_image)
            #     if point_coords is not None:
            #         assert input_labels is not None
            #         show_points(point_coords, input_labels, plt.gca())
            #     if box_coords is not None:
            #         # boxes
            #         show_box(box_coords, plt.gca())
            #     if len(scores) > 1:
            #         plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis("off")
            plt.savefig(save_path, dpi=200)
            plt.close()
