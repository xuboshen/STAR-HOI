import argparse
import logging
import os
import time

import cv2
import decord
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from hoid_model.utils.config import cfg
from PIL import Image

from star_hoi.data.dataloader import build_dataloader, build_detection_test_loader
from star_hoi.data.dataset import build_dataset
from star_hoi.data.sampler import InferenceSampler
from star_hoi.data.utils import get_frame_ids
from star_hoi.engine import validate_image, validate_video
from star_hoi.evaluation.evaluator import build_evaluator
from star_hoi.model.model import build_model
from star_hoi.utils.argparse_utils import parse_args
from star_hoi.utils.utils import redirect_output, save_args

logging.getLogger().setLevel(logging.INFO)

if __name__ == "__main__":
    args = parse_args()
    # path preparation
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.vis_path, exist_ok=True)
    save_args(args)
    redirect_output(os.path.join(args.output_path, "outputs.log"))
    if args.multigpu_inference:
        accelerator = Accelerator()
    # data preparation
    if args.dataset_name == "visor_image":
        # register for hos evaluation
        model_names = ["hoid", "sam_image"]
        val_dataset = build_dataset(
            args, args.dataset_name, args.anno_path, args.image_path
        )
        sampler = InferenceSampler(len(val_dataset))
        val_loader = build_detection_test_loader(val_dataset, sampler)
        # evaluator
        evaluator = build_evaluator(args, args.eval_task, args.output_path)
    elif args.dataset_name == "ego4d_video":
        model_names = ["hoid", "sam_video"]
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
    # demos
    elif args.dataset_name == "demo_image":
        model_names = ["hoid", "sam_image"]
        image = Image.open("examples/ego4d_image_vis/ego4d_example_raw.png")
        image = np.array(image.convert("RGB"))  # H, W, 3
    elif args.dataset_name == "demo_video":
        model_names = ["hoid", "sam_video"]
        vr = decord.VideoReader("examples/ego4d_video_vis/ego4d_example.mp4")
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
    else:
        raise NotImplementedError(f"{args.dataset_name} is not implemented yet")
    # model preparation
    hoi_detector, sam_model = build_model(args, model_names)
    if args.multigpu_inference:
        hoi_detector = hoi_detector.to(accelerator.device)
        sam_model = sam_model.to(accelerator.device)
        hoi_detector, sam_model, val_loader = accelerator.prepare(
            hoi_detector, sam_model, val_loader
        )
        accelerator.wait_for_everyone()
    # inference
    if args.dataset_name == "visor_image":
        validate_image.validate_visor_image(
            args,
            val_loader,
            hoi_detector,
            sam_model,
            evaluator,
            args.hand_prompt_type,
            args.obj_prompt_type,
        )
    elif args.dataset_name == "ego4d_video":
        validate_video.validate_ego4d_video(
            args,
            val_loader,
            hoi_detector,
            sam_model,
            args.hand_prompt_type,
            args.obj_prompt_type,
        )
    elif args.dataset_name == "demo_image":
        validate_image.validate_demo_image(
            args,
            [image],
            hoi_detector,
            sam_model,
            args.hand_prompt_type,
            args.obj_prompt_type,
        )
    elif args.dataset_name == "demo_video":
        validate_video.validate_demo_video(
            args,
            [video_frames],
            hoi_detector,
            sam_model,
            args.hand_prompt_type,
            args.obj_prompt_type,
        )
