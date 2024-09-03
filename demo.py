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
from detectron2.structures import Boxes
from hoid_model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from hoid_model.utils.net_utils import (  # (1) here add a function to viz
    load_net,
    save_net,
    vis_detections,
    vis_detections_filtered_objects,
    vis_detections_filtered_objects_PIL,
    vis_detections_PIL,
)
from PIL import Image
from tqdm import tqdm

from star_hoi.data.dataloader import build_detection_test_loader
from star_hoi.data.dataset import build_dataset
from star_hoi.data.instances import Instances
from star_hoi.data.sampler import InferenceSampler
from star_hoi.data.utils import (
    concate_hoi,
    detection_post_process,
    initialize_inputs,
    prepare_boxes,
    prepare_hoid_inputs,
    prepare_points,
    prepare_sam_image_inputs,
)
from star_hoi.evaluation.evaluator import build_evaluator
from star_hoi.model.model import build_model
from star_hoi.utils.utils import numpy_to_torch_dtype, redirect_output, save_args
from star_hoi.utils.visualize import show_masks

logging.getLogger().setLevel(logging.INFO)


def prepare_output_for_evaluator(
    image_shape, masks, boxes, scores, hand_dets, obj_dets
):
    """
    explanations of detailed params: https://github.com/epic-kitchens/epic-kitchens-100-hand-object-bboxes/blob/af22bca2124389b96fbf01b19f8684c302fea22f/src/raw_detections/types.py#L24
    sequence: objects+hands
    args:
        image_shape: (height, width), (750, 1333)
        masks: N (num_of_instances), height, width
        boxes: N, 2
        scores: sam scores, N, 1
        handsides: pred_handsides, N, 2. 0: left hand, 1: right hand
        hand_dets/obj_dets: note the case that they are <None>
    """
    output = {}
    result = Instances(image_shape)

    classes = []
    # 0: hand, 1: object
    if obj_dets is not None:
        classes.extend([1] * obj_dets.shape[0])
    if hand_dets is not None:
        classes.extend([0] * hand_dets.shape[0])
    if obj_dets is None and hand_dets is None:
        classes = [1]

    # handside convertion
    handsides = concate_hoi(
        hand_dets, obj_dets, -1
    )  # np.concatenate([obj_dets[:, -1], hand_dets[:, -1]])
    # convert to N, 2, one-hots
    one_hot_mat = np.eye(2)
    handsides = one_hot_mat[handsides.astype(int)].astype(float)
    # contact convertion
    contacts = concate_hoi(
        hand_dets, obj_dets, 5
    )  # np.concatenate([obj_dets[:, 5], hand_dets[:, 5]])
    # map [1, 4] to 1, while 0 remains 0, also to one-hot
    contacts = np.where(contacts == 0, 0, 1)
    contacts = one_hot_mat[contacts.astype(int)].astype(float)
    # offsets convertion
    # TBD: seems not very correct in scales.
    offsets = concate_hoi(
        hand_dets, obj_dets, list(range(6, 9))
    )  # np.concatenate([obj_dets[:, 6:9], hand_dets[:, 6:9]])
    # classes convertion
    pred_classes = torch.tensor(classes, dtype=int)
    # TBD: scores to be considered
    result.scores = scores.reshape(-1)
    result.pred_masks = masks.astype(bool)
    result.pred_boxes = Boxes(torch.tensor(boxes, dtype=numpy_to_torch_dtype(boxes)))
    result.pred_handsides = handsides
    result.pred_classes = pred_classes
    result.pred_contacts = contacts
    result.pred_offsets = offsets

    for key, value in result._fields.items():
        if key != "pred_boxes" and isinstance(value, np.ndarray):
            result._fields[key] = torch.tensor(
                value, dtype=numpy_to_torch_dtype(value.dtype)
            )
    output["instances"] = result

    return output


def sam_prediction(sam_model, image, prompt_input_list):
    """
    sam prediction: input images, output region masks
    args:
        prompt_list: [obj, hand]
    returns:
        masks: N, 1, H, W or 1, H, W. obj_masks->hand_masks
        scores: N, 1 or N
    """
    mask_list = []
    score_list = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        sam_model.set_image(image)
        for prompt_input in prompt_input_list:
            masks, scores, _ = sam_model.predict(**prompt_input)
            if len(masks.shape) == 4:
                mask_list.extend([msk for msk in masks])
                score_list.extend([sco for sco in scores])
            else:
                mask_list.append(masks)
                score_list.append(scores)
    _, H, W = mask_list[0].shape
    if len(mask_list) > 1:
        masks = np.stack(mask_list)
        scores = np.stack(score_list)
    else:
        masks = np.array(mask_list).reshape(1, H, W)
        scores = np.array(score_list).reshape(-1)

    return masks, scores


@torch.no_grad()
def validate_visor_image(
    args,
    val_loader,
    hoi_detector,
    sam_model,
    evaluator,
    hand_prompt_type,
    obj_prompt_type,
    use_half=False,
):
    """
    used for inference, i.e., with evaluation
    """
    print("Inference Begins...")
    hoi_detector.eval()
    is_multi_obj = args.multiobj_track
    hoid_test_short_size = (args.hoid_test_short_size,)
    input_dicts = initialize_inputs(no_cuda=args.no_cuda)
    # evaluator init
    evaluator.reset()

    for i, inputs in tqdm(enumerate(val_loader), total=len(val_loader)):
        # hoi_detector processing
        # from BGR, 750, 1333-->750, 1333, BGR
        im_hoi = inputs[0]["image"].permute(1, 2, 0).numpy()
        # from BGR to RGB, for the sake of sam2 input
        image = np.ascontiguousarray(im_hoi[..., ::-1])
        input_dicts, im_scales = prepare_hoid_inputs(
            im_hoi, hoid_test_short_size, args.hoid_test_max_size, **input_dicts
        )
        # hoid model inference
        (rois, cls_prob, bbox_pred, _, _, _, _, _, loss_list) = hoi_detector(
            **input_dicts
        )
        hand_dets, obj_dets = detection_post_process(
            args,
            rois,
            cls_prob,
            bbox_pred,
            loss_list,
            im_scales,
            input_dicts["im_info"],
            args.thresh_hoid_score,
        )
        # hoid visualization
        if args.vis and i % args.vis_freq == 0:
            # vis: raw image
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            plt.axis("off")
            plt.savefig(f"{args.vis_path}/test_visor_example_{i}.png", dpi=200)
            plt.close()

            # vis: image with boxes
            im2show = np.copy(im_hoi)
            im2show = vis_detections_filtered_objects_PIL(
                im2show,
                obj_dets,
                hand_dets,
                args.thresh_hoid_score,
                args.thresh_hoid_score,
                font_path="hand_object_detector/lib/hoid_model/utils/times_b.ttf",
            )
            im2show.save(os.path.join(args.vis_path, f"test_visor_example_det_{i}.png"))
        # sam image processing
        prompt_list = prepare_sam_image_inputs(
            args,
            hand_dets,
            obj_dets,
            is_multi_obj,
            args.multimask_output,
            hand_prompt_type,
            obj_prompt_type,
        )
        # sam inference
        masks, sam_scores = sam_prediction(sam_model, image, prompt_list)
        # fix box_input if exists None
        box_input = prepare_boxes(hand_dets, obj_dets, is_multi_obj, "hoi")
        if box_input is None:
            box_input = np.zeros([1, 4])
            masks = np.zeros_like(masks, dtype=masks.dtype)
        # sam visualization
        if args.vis and i % args.vis_freq == 0:
            show_masks(
                image,
                masks,
                sam_scores,
                box_coords=box_input,
                thresh_sam_score=args.thresh_sam_score,
                save_path=os.path.join(
                    args.vis_path, f"test_visor_example_mask_{i}.png"
                ),
            )

        # evaluator run
        outputs = prepare_output_for_evaluator(
            image_shape=image.shape[:2],
            masks=masks.squeeze(1) if masks.shape[1] == 1 else masks,
            boxes=box_input,
            scores=sam_scores.squeeze(1) if sam_scores.shape == 2 else sam_scores,
            hand_dets=hand_dets,
            obj_dets=obj_dets,
        )
        evaluator.process(inputs, [outputs])
        if args.debug and i == 17:
            return
    results = evaluator.evaluate()
    if results is None:
        results = {}

    return results


@torch.no_grad()
def validate_demo_video(
    args,
    val_loader,
    hoi_detector,
    sam_model,
    hand_prompt_type,
    obj_prompt_type,
    use_half=False,
):
    pass


@torch.no_grad()
def validate_demo_image(
    args,
    val_loader,
    hoi_detector,
    sam_model,
    hand_prompt_type,
    obj_prompt_type,
    use_half=False,
):
    """
    only used for demo, i.e. without evaluation
    here, val_loader = [image]
    """
    print("demo image processing begins...")
    hoi_detector.eval()
    is_multi_obj = args.multiobj_track
    tgt_type = args.target_type
    hoid_test_short_size = (args.hoid_test_short_size,)
    input_dicts = initialize_inputs(no_cuda=args.no_cuda)
    # im_hoi = cv2.imread("examples/ego4d_example.png")
    for i, image in enumerate(val_loader):
        im_hoi = image[..., ::-1]
        # hoi_detector pre-processing
        input_dicts, im_scales = prepare_hoid_inputs(
            im_hoi, hoid_test_short_size, args.hoid_test_max_size, **input_dicts
        )  # input: BGR
        # model inference
        (rois, cls_prob, bbox_pred, _, _, _, _, _, loss_list) = hoi_detector(
            **input_dicts
        )
        hand_dets, obj_dets = detection_post_process(
            args,
            rois,
            cls_prob,
            bbox_pred,
            loss_list,
            im_scales,
            input_dicts["im_info"],
            args.thresh_hoid_score,
        )
        if args.vis and i % args.vis_freq == 0:
            # vis: raw image
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            plt.axis("off")
            plt.savefig(f"{args.vis_path}/ego4d_example.png", dpi=200)
            plt.close()

            im2show = np.copy(im_hoi)
            im2show = vis_detections_filtered_objects_PIL(
                im2show,
                obj_dets,
                hand_dets,
                args.thresh_hoid_score,
                args.thresh_hoid_score,
                font_path="hand_object_detector/lib/hoid_model/utils/times_b.ttf",
            )
            im2show.save(os.path.join(args.vis_path, "ego4d_example_det.png"))
            print(f"saving hoi detection image ... to {args.vis_path}/ego4d_det.png")
        # sam pre-processing
        if hand_prompt_type == "box" and obj_prompt_type == "box":
            box_input = prepare_boxes(hand_dets, obj_dets, is_multi_obj, tgt_type)
            prompt_inputs = dict(box=box_input, multimask_output=args.multimask_output)
        else:
            raise NotImplementedError(
                f"{hand_prompt_type} or {args.obj_prompt_type} error"
            )
        # sam inference
        masks, sam_scores = sam_prediction(sam_model, image, [prompt_inputs])
        # sam image visualizations
        if args.vis and i % args.vis_freq == 0:
            show_masks(
                image,
                masks,
                sam_scores,
                box_coords=box_input,
                thresh_sam_score=args.thresh_sam_score,
                save_path=os.path.join(args.vis_path, "ego4d_example_mask.png"),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--foo", help="foo help")
    # common arguments
    parser.add_argument(
        "--image-path",
        type=str,
        default="images/ego4d_example.png",
        help="which is also the image_root",
    )
    parser.add_argument(
        "--anno-path",
        type=str,
        default="VISOR/epick_visor_coco_hos/annotations/val.json",
        help="the annotation file",
    )
    parser.add_argument(
        "--vis",
        help="default is False, test the results",
        action="store_true",
    )
    parser.add_argument(
        "--vis-freq",
        type=int,
        default=100,
        help="the visualization save frequency",
    )
    parser.add_argument(
        "--debug",
        help="debug model, default is false",
        action="store_true",
    )
    parser.add_argument(
        "--dataset-name",
        default="demo_image",
        type=str,
        choices=["demo_image", "demo_video", "visor_image", "visor_video"],
        help="choose from visor_sparse, visor_dense, demo_image, and demo_video",
    )
    parser.add_argument(
        "--print-freq",
        default=100,
        type=int,
        help="print frequency during inference",
    )
    parser.add_argument(
        "--model-names",
        default=["hoid", "sam"],
        type=list,
        help="use of off-the-shelf models",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Default is False, i.e., with cuda",
    )
    parser.add_argument(
        "--test-sam2",
        nargs="*",
        help="support more than one argument and merge into a list.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="outputs/demo",
        help="save to output path",
    )
    parser.add_argument(
        "--vis-path",
        type=str,
        default="image_vis",
        help="visualization path to save",
    )

    # hoid model parameters
    parser.add_argument(
        "--hoid-test-max-size",
        type=int,
        default=1200,
        help="the largest width of input image size to hoi detector model",
    )
    parser.add_argument(
        "--hoid-test-short-size",
        type=int,
        default=800,
        help="the shortest length of input image size to hoi detector model",
    )
    parser.add_argument(
        "--thresh-hoid-score",
        type=float,
        default=0.3,
        help="the threshold score that a detection of hoi detector is good",
    )
    parser.add_argument(
        "--target-type",
        type=str,
        default="hoi",
        choices=["obj", "lh", "rh", "hoi"],
        help="choose tracking targets after hoi detection",
    )
    parser.add_argument(
        "--hoid-model",
        type=str,
        default="res101",
        choices=["vgg16", "res50", "res101", "res152"],
        help="types of fasterRCNN",
    )
    parser.add_argument(
        "--hoid-test-nms",
        type=float,
        default=0.3,
        help="test nms score",
    )
    parser.add_argument(
        "--hoid-checkpoint",
        type=str,
        default="checkpoints/faster_rcnn_1_8_132028.pth",
        help="path to the hoid checkpoint",
    )
    parser.add_argument(
        "--cag",
        dest="class_agnostic",
        help="whether perform class_agnostic bbox regression, default is False",
        action="store_true",
    )
    # sam2 model parameters
    parser.add_argument(
        "--thresh-sam-score",
        type=float,
        default=0.7,
        help="the threshold score that a object is good",
    )
    parser.add_argument(
        "--multiobj-track",
        action="store_true",
        help="Default to be False, i.e. default is single_object",
    )
    parser.add_argument(
        "--hand-prompt-type",
        default="box",
        choices=["box", "point", "mask", "mixed"],
        help="prompt for initial frame, the hand prompt type for sam2.",
    )
    parser.add_argument(
        "--obj-prompt-type",
        type=str,
        default="box",
        choices=["box", "point", "mask", "mixed"],
        help="prompt for initial frame, the object prompt type for sam2 input",
    )
    parser.add_argument(
        "--sam-model-cfg",
        type=str,
        default="sam2_hiera_l.yaml",
        choices=[
            "sam2_hiera_t.yaml",
            "sam2_hiera_s.yaml",
            "sam2_hiera_b+.yaml",
            "sam2_hiera_l.yaml",
        ],
        help="model config of sam model",
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=str,
        default="checkpoints/sam2_hiera_large.pt",
        help="path to the sam checkpoint",
    )
    parser.add_argument(
        "--multimask-output",
        action="store_true",
        help="Default is False, i.e., only one mask output, which is suitable for non-ambiguous prompts (e.g., click + box)",
    )
    # Downstream datasets
    parser.add_argument(
        "--eval-task",
        type=str,
        default="hand_obj",
        choices=["hand_obj"],
        help="Used for eval_task of single-image VISOR-HOS Benchmark",
    )

    args = parser.parse_args()
    model_names = args.model_names
    # path preparation
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.vis_path, exist_ok=True)
    save_args(args)
    redirect_output(os.path.join(args.output_path, "outputs.log"))
    # data preparation
    if args.dataset_name.startswith("visor"):
        # register for hos evaluation
        val_dataset = build_dataset(
            args, args.dataset_name, args.anno_path, args.image_path
        )
        sampler = InferenceSampler(len(val_dataset))
        val_loader = build_detection_test_loader(val_dataset, sampler)
        # evaluator
        evaluator = build_evaluator(args, args.eval_task, args.output_path)
    elif args.dataset_name == "demo_image":
        image = Image.open("examples/ego4d_example.png")
        image = np.array(image.convert("RGB"))
    elif args.dataset_name == "demo_video":
        video = decord.VideoReader("./output_video.mp4")
    else:
        raise NotImplementedError(f"{args.dataset_name} is not implemented yet")
    # model preparation
    hoi_detector, sam_model = build_model(args, model_names)

    # inference
    if args.dataset_name.startswith("visor_image"):
        validate_visor_image(
            args,
            val_loader,
            hoi_detector,
            sam_model,
            evaluator,
            args.hand_prompt_type,
            args.obj_prompt_type,
        )
    elif args.dataset_name == "demo_image":
        validate_demo_image(
            args,
            [image],
            hoi_detector,
            sam_model,
            args.hand_prompt_type,
            args.obj_prompt_type,
        )
    elif args.dataset_name == "demo_video":
        validate_demo_video(
            args,
            [video],
            hoi_detector,
            sam_model,
            args.hand_prompt_type,
            args.obj_prompt_type,
        )
    # evaluate()
