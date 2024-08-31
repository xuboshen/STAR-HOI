import argparse
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from detectron2.structures import Boxes
from hoid_model.faster_rcnn.resnet import resnet
from hoid_model.faster_rcnn.vgg16 import vgg16
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
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from star_hoi.data.dataloader import build_detection_test_loader
from star_hoi.data.dataset import VISOR_Image, build_dataset
from star_hoi.data.instances import Instances
from star_hoi.data.sampler import InferenceSampler
from star_hoi.data.utils import (
    detection_post_process,
    initialize_inputs,
    prepare_boxes,
    prepare_inputs,
)
from star_hoi.evaluation.evaluator import build_evaluator
from star_hoi.utils.utils import concate_hoi, numpy_to_torch_dtype
from star_hoi.utils.visualize import show_masks


# # from star_hoi.utils.metrics import calculate_visor_metrics
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
    if obj_dets is not None:
        classes.extend([1] * obj_dets.shape[0])
    if hand_dets is not None:
        classes.extend([0] * hand_dets.shape[0])

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
    pred_classes = torch.tensor(classes, dtype=bool)
    # TBD: scores to be considered
    result.scores = scores
    result.pred_masks = masks
    result.pred_boxes = Boxes(torch.tensor(boxes, dtype=numpy_to_torch_dtype(boxes)))
    result.pred_handsides = handsides
    result.pred_classes = pred_classes
    result.pred_contacts = contacts
    result.pred_offsets = offsets

    for key, value in result._fields.items():
        try:
            result._fields[key] = torch.tensor(value, dtype=numpy_to_torch_dtype(value))
        except Exception as E:
            print(E)
    output["instances"] = result

    return output


def get_hoid_model(args):
    # load model
    load_name = args.hoid_checkpoint
    pascal_classes = np.asarray(["__background__", "targetobject", "hand"])
    # initilize the network here.
    if args.hoid_model == "vgg16":
        fasterRCNN = vgg16(
            pascal_classes, pretrained=False, class_agnostic=args.class_agnostic
        )
    elif args.hoid_model == "res101":
        fasterRCNN = resnet(
            pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic
        )
    elif args.hoid_model == "res50":
        fasterRCNN = resnet(
            pascal_classes, 50, pretrained=False, class_agnostic=args.class_agnostic
        )
    elif args.hoid_model == "res152":
        fasterRCNN = resnet(
            pascal_classes, 152, pretrained=False, class_agnostic=args.class_agnostic
        )
    else:
        print("network is not defined")
        raise NotImplementedError(f"{args.hoid_model} not implemented yet.")

    fasterRCNN.create_architecture()

    print("load checkpoint %s" % (load_name))
    if args.no_cuda is False:
        checkpoint = torch.load(load_name)
    else:
        checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
    fasterRCNN.load_state_dict(checkpoint["model"])
    print("load hoi detector model successfully!")
    if args.no_cuda is False:
        fasterRCNN = fasterRCNN.cuda()

    return fasterRCNN


def sam_prediction(sam_model, image, prompt_inputs: dict):
    """
    sam prediction: input images, output region masks
    """
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        sam_model.set_image(image)
        # masks, _, _ = predictor.predict()
        masks, scores, _ = sam_model.predict(**prompt_inputs)
        # show_masks(
        #     image,
        #     masks,
        #     scores,
        #     box_coords=box_input,
        #     thresh_sam_score=args.thresh_sam_score,
        # )
        # print(masks.shape)
    return masks, scores


def get_model(args, model_names):
    for model_name in model_names:
        if model_name == "hoid":
            hoi_detector = get_hoid_model(args)
        elif model_name == "sam":
            sam_model = SAM2ImagePredictor(
                build_sam2(args.sam_model_cfg, args.sam_checkpoint)
            )
        else:
            raise NotImplementedError(f"{model_name} Not implemented yet")

    return hoi_detector, sam_model


def validate_visor_image(
    args, val_loader, hoi_detector, sam_model, evaluator, prompt_type, use_half=False
):
    """
    used for inference, i.e., with evaluation
    """
    print("Inference Begins...")
    hoi_detector.eval()
    is_multi_obj = args.multiobj_track
    tgt_type = args.target_type
    input_dicts = initialize_inputs(no_cuda=args.no_cuda)
    # evaluator init
    evaluator.reset()

    for i, inputs in enumerate(val_loader):
        # hoi_detector pre-processing
        # from BGR, 750, 1333-->750, 1333, BGR
        im_hoi = inputs[0]["image"].permute(1, 2, 0).numpy()
        # from BGR to RGB, for the sake of sam2 input
        image = np.ascontiguousarray(im_hoi[..., ::-1])
        if args.vis:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            plt.axis("off")
            plt.savefig(f"image_vis/test_visor_example_{i}.png", dpi=200)

        input_dicts, im_scales = prepare_inputs(im_hoi, **input_dicts)
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
        if args.vis:
            im2show = np.copy(im_hoi)
            im2show = vis_detections_filtered_objects_PIL(
                im2show,
                obj_dets,
                hand_dets,
                0.5,
                0.5,
                font_path="hand_object_detector/lib/hoid_model/utils/times_b.ttf",
            )
            im2show.save(os.path.join(args.vis_path, f"test_visor_example_det_{i}.png"))
        # sam pre-processing
        if prompt_type == "box":
            box_input = prepare_boxes(hand_dets, obj_dets, is_multi_obj, tgt_type)
            prompt_inputs = dict(box=box_input, multimask_output=args.multimask_output)
        else:
            raise NotImplementedError(f"{args.prompt_type} not implemented yet")
        # model inference
        masks, sam_scores = sam_prediction(sam_model, image, prompt_inputs)
        if args.vis:
            show_masks(
                image,
                masks,
                sam_scores,
                box_coords=box_input,
                thresh_sam_score=args.thresh_sam_score,
                save_fig_name=f"test_visor_example_mask_{i}",
            )
            print("mask.shape:", masks.shape)
            print("sam_scores.shape:", sam_scores.shape)
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
    results = evaluator.evaluate()

    if results is None:
        results = {}

    return results


def validate_demo_image(
    args, val_loader, hoi_detector, sam_model, prompt_type, use_half=False
):
    """
    only used for demo, i.e. without evaluation
    here, val_loader = [image]
    """
    print("demo image processing begins...")
    hoi_detector.eval()
    is_multi_obj = args.multiobj_track
    tgt_type = args.target_type
    input_dicts = initialize_inputs(no_cuda=args.no_cuda)
    # im_hoi = cv2.imread("examples/ego4d_example.png")
    for i, image in enumerate(val_loader):
        im_hoi = image[..., ::-1]
        # hoi_detector pre-processing
        input_dicts, im_scales = prepare_inputs(im_hoi, **input_dicts)  # input: BGR
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
        if args.vis:
            im2show = np.copy(im_hoi)
            im2show = vis_detections_filtered_objects_PIL(
                im2show, obj_dets, hand_dets, 0.5, 0.5
            )
            im2show.save(os.path.join(args.vis_path, "ego4d_det_det.png"))
            print(f"saving hoi detection image ... to {args.vis_path}/ego4d_det.png")
        # sam pre-processing
        if prompt_type == "box":
            box_input = prepare_boxes(hand_dets, obj_dets, is_multi_obj, tgt_type)
            prompt_inputs = dict(box=box_input, multimask_output=args.multimask_output)
        else:
            raise NotImplementedError(f"{args.prompt_type} not implemented yet")
        # model inference
        masks, sam_scores = sam_prediction(sam_model, image, prompt_inputs)
        if args.vis:
            show_masks(
                image,
                masks,
                sam_scores,
                box_coords=box_input,
                thresh_sam_score=args.thresh_sam_score,
            )
            print(masks.shape)


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
        "--debug",
        help="debug model, default is false",
        action="store_true",
    )
    parser.add_argument(
        "--dataset-name",
        default="demo",
        type=str,
        choices=["demo", "visor_image", "visor_video"],
        help="choose from visor_sparse, visor_dense, demo",
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
        default="outputs",
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
        "--thresh-hoid-score",
        type=float,
        default=0.5,
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
        "--prompt-type",
        type=str,
        default="box",
        choices=["box", "point", "mask", "mixed"],
        help="the threshold score that a object is good",
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

    # data preparation
    if args.dataset_name.startswith("visor"):
        val_dataset = build_dataset(
            args, args.dataset_name, args.anno_path, args.image_path
        )
        sampler = InferenceSampler(len(val_dataset))
        val_loader = build_detection_test_loader(val_dataset, sampler)
        # evaluator
        evaluator = build_evaluator(args, args.eval_task)
    elif args.dataset_name == "demo":
        image = Image.open("examples/ego4d_example.png")
        image = np.array(image.convert("RGB"))
    # model preparation
    hoi_detector, sam_model = get_model(args, model_names)

    # inference
    if args.dataset_name.startswith("visor_image"):
        if args.debug:
            validate_visor_image(
                args, val_loader, hoi_detector, sam_model, evaluator, args.prompt_type
            )
        else:
            raise NotImplementedError("Not ready yet, still in debug states")
    elif args.dataset_name == "demo":
        validate_demo_image(args, [image], hoi_detector, sam_model, args.prompt_type)

    # evaluate()
