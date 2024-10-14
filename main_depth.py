import argparse
import glob
import os

import cv2
import decord
import matplotlib
import numpy as np
import torch
from tqdm import tqdm

from depth_anything_v2.depth_anything_v2.dpt import DepthAnythingV2
from star_hoi.data.dataloader import build_dataloader, build_detection_test_loader
from star_hoi.data.dataset import build_dataset
from star_hoi.data.sampler import InferenceSampler
from star_hoi.data.utils import get_frame_ids
from star_hoi.utils.argparse_utils import parse_args
from star_hoi.utils.utils import redirect_output, save_args


def ego4d_video_inference(val_loader, depth_anything, args):
    for i, inputs in tqdm(enumerate(val_loader), total=len(val_loader)):
        frames, video_uid, narration_time = (
            inputs[0]["frame"],
            inputs[0]["uid"],
            inputs[0]["timestamp"],
        )
        if args.save_results and os.path.exists(
            os.path.join(args.output_path, video_uid, narration_time + "_depth.npz")
        ):
            continue
        if frames is None or frames.sum() == 0:
            continue
        os.makedirs(os.path.join(args.output_path, video_uid), exist_ok=True)
        depth_list = []
        for i, frame in enumerate(frames):
            frame = frame.numpy()
            frame = frame.astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            depth = depth_anything.infer_image(frame, args.depth_input_size)

            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)
            from star_hoi.utils.utils import resize_with_aspect_ratio

            resized_depth = resize_with_aspect_ratio(depth, 288)
            depth_list.append(resized_depth)
            # Save or process the depth output
            output_frame_path = os.path.join(
                args.output_path, video_uid, f"frame_{i}.png"
            )
            cv2.imwrite(output_frame_path, resized_depth)
            print(f"Saved depth frame {i} at {output_frame_path}")
        depth_array = np.array(depth_list)
        np.savez_compressed(
            os.path.join(args.output_path, video_uid, f"{narration_time}_depth.npz"),
            images=depth_array,
        )
        break


def ego4d_demo_inference(video_frames, depth_anything, args):
    # Iterate through the video frames and perform depth inference
    for i, frames in enumerate(tqdm(video_frames)):
        depth = depth_anything.infer_image(frames, args.depth_input_size)

        # Normalize the depth values to [0, 255]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)

        # Convert grayscale depth to a 3-channel image
        depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)

        # Save or process the depth output
        output_frame_path = os.path.join(args.output_path, f"frame_{i}.png")
        cv2.imwrite(output_frame_path, depth)
        print(f"Saved depth frame {i} at {output_frame_path}")


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    save_args(args)
    redirect_output(os.path.join(args.output_path, "outputs.log"))

    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # model preparation
    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {
            "encoder": "vitb",
            "features": 128,
            "out_channels": [96, 192, 384, 768],
        },
        "vitl": {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        },
        "vitg": {
            "encoder": "vitg",
            "features": 384,
            "out_channels": [1536, 1536, 1536, 1536],
        },
    }

    depth_anything = DepthAnythingV2(**model_configs[args.depth_encoder])
    depth_anything.load_state_dict(
        torch.load(
            f"checkpoints/depth_anything_v2_{args.depth_encoder}.pth",
            map_location="cpu",
        )
    )
    depth_anything = depth_anything.to(DEVICE).eval()

    # dataset preparation
    if args.dataset_name == "demo_video":
        model_names = ["depth_anything"]
        vr = decord.VideoReader("examples/ego4d_example.mp4")
        fps = vr.get_avg_fps()
        start_second = 0
        clip_length = args.clip_length
        frame_offset = int(np.round(start_second * fps))
        total_duration = len(vr)
        frame_ids = get_frame_ids(
            frame_offset,
            min(frame_offset + total_duration, len(vr)),
            num_segments=clip_length,
        )
        video_frames = vr.get_batch(frame_ids).asnumpy()  # B, H, W, 3
    elif args.dataset_name == "ego4d_video":
        val_dataset = build_dataset(
            args, args.dataset_name, args.anno_path, args.image_path
        )
        sampler = InferenceSampler(len(val_dataset))
        val_loader = build_dataloader(
            args.dataset_name,
            val_dataset,
            batch_size=1,
            sampler=sampler,
            drop_last=False,
            num_workers=args.num_workers,
        )

    margin_width = 50
    cmap = matplotlib.colormaps.get_cmap("Spectral_r")

    if args.dataset_name == "ego4d_video":
        ego4d_video_inference(val_loader, depth_anything, args)
    elif args.dataset_name == "demo_video":
        ego4d_demo_inference(video_frames, depth_anything, args)
