import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

matplotlib.use("Agg")


def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
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


def show_points():
    pass


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
):
    if len(box_coords.shape) == 1:

        for i, (mask, score) in enumerate(zip(masks, scores)):
            print(i)
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
            os.makedirs("images_vis", exist_ok=True)
            plt.savefig(f"images_vis/masked_ego4d_example_{i}.png", dpi=200)
    elif len(box_coords.shape) == 2:
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        if len(masks.shape) == 4:  # N_prompt, Number_of_images, H, W
            masks = np.transpose(masks, (1, 0, 2, 3))
            scores = np.transpose(scores, (1, 0))
        for i, (mask, score) in enumerate(zip(masks, scores)):
            print(i)
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
            os.makedirs("images_vis", exist_ok=True)
            plt.savefig(f"images_vis/masked_ego4d_example_{i}.png", dpi=200)


"""
hyperparameters:
    multimask_output:
        True: 3 masks and select one, True if vague prompts, e.g., a single click; We can select best mask by quality score
        False: only one output; For non-ambiguous prompts, e.g., multiple input prompts.
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--foo", help="foo help")
    # parser.add_argument('--image-dir', type=str, choices=range(1, 4))
    parser.add_argument("--image-dir", type=str, default="images/ego4d_example.png")
    parser.add_argument(
        "--multiobj-track",
        action="store_true",
        help="Default to be False, i.e. default is single_object",
    )
    parser.add_argument(
        "--thresh-sam-score",
        type=float,
        default=0.7,
        help="the threshold score that a object is good",
    )
    parser.add_argument(
        "--prompt-type",
        type=str,
        default="box",
        choices=["box", "point", "mask", "mixed"],
        help="the threshold score that a object is good",
    )
    parser.add_argument(
        "--multimask-output",
        action="store_true",
        help="Default is False, i.e., only one mask output, which is suitable for non-ambiguous prompts (e.g., click + box)",
    )
    parser.add_argument(
        "--test-sam2",
        nargs="*",
        help="support more than one argument and merge into a list.",
    )

    args = parser.parse_args()

    obj_boxes = np.array(
        [
            [489.60413, 523.0943, 776.3056, 724.338],
            [909.6571, 461.0867, 969.9202, 526.4396],
        ]
    )
    hand_boxes = np.array(
        [
            [432.4632, 658.10315, 523.91534, 774.21014],
            [902.05975, 484.67023, 981.31226, 602.2021],
        ]
    )
    # input_point = np.array([[500, 375], [1125, 625]])

    # open image with Image
    image = Image.open(args.image_dir)
    image = np.array(image.convert("RGB"))

    multiple_object_track = args.multiobj_track
    if args.prompt_type == "box":
        if multiple_object_track:
            # default
            box_input = np.concatenate([obj_boxes, hand_boxes], axis=0)  # (N, 4)
        else:
            box_input = hand_boxes[0]  # (4, )
        print(box_input.shape)

        input_args_to_sam2 = dict(box=box_input, multimask_output=args.multimask_output)
    else:
        raise NotImplementedError(f"{args.prompt_type} is not implemented yet.")

    checkpoint = "../checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image)
        # masks, _, _ = predictor.predict()
        masks, scores, _ = predictor.predict(**input_args_to_sam2)
        show_masks(
            image,
            masks,
            scores,
            box_coords=box_input,
            thresh_sam_score=args.thresh_sam_score,
        )
        print(masks.shape)
