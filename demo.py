import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from hoid_model.faster_rcnn.resnet import resnet
from hoid_model.faster_rcnn.vgg16 import vgg16
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from star_hoi.data.dataset import VISOR
from star_hoi.data.utils import (
    detection_post_process,
    initialize_inputs,
    prepare_boxes,
    prepare_inputs,
)
from star_hoi.utils.metrics import calculate_visor_metrics


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
    if args.cuda is True:
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


def validate_visor(args, val_loader, hoi_detector, sam_model, use_half=False):
    hoi_detector.eval()
    sam_model.eval()
    # if use_half:
    #     model.half()
    # with torch.no_grad():
    #     print('=> start forwarding')
    #     all_preds = []
    #     all_gts = []
    #     all_types = []
    #     end_time = time.time()
    #     for i, inputs in enumerate(val_loader):
    #         if i % args.print_freq == 0:
    #             print('finish batch {}/{} in {} sec'.format(i, len(val_loader), time.time() - end_time))
    #             end_time = time.time()
    #         texts_query = inputs[0].cuda(non_blocking=True)
    #         frames = inputs[1].cuda(non_blocking=True)
    #         if use_half:
    #             frames = frames.half()
    #         verb_choice = inputs[2].cuda(non_blocking=True)
    #         noun_choice = inputs[3].cuda(non_blocking=True)
    #         # answer = inputs[3]
    #         # q_type = inputs[4]
    #         if len(inputs) == 7:
    #             masks_query = inputs[5].cuda(non_blocking=True)
    #         else:
    #             masks_query = None

    #         batch_size = frames.shape[0]

    #         frames_options =  frames #frames_options.view(-1, *frames_options.shape[2:])
    #         image_features = dist_utils.get_model(model).encode_image(frames_options)
    #         # image_features = image_features.view(batch_size, -1, *image_features.shape[1:])

    #         if masks_query is not None:
    #             query_features = dist_utils.get_model(model).encode_text(texts_query, attention_mask=masks_query)
    #         else:
    #             query_features = dist_utils.get_model(model).encode_text(texts_query)

    #             verb_features = dist_utils.get_model(model).encode_text(verb_choice.view(-1,verb_choice.shape[-1]))
    #             verb_features = verb_features.view(batch_size, -1, verb_features.shape[-1])
    #             noun_features = dist_utils.get_model(model).encode_text(noun_choice.view(-1,noun_choice.shape[-1]))
    #             noun_features = noun_features.view(batch_size, -1, noun_features.shape[-1])

    #         # all_gts.append(answer)
    #         # all_types.append(q_type)
    #         for j in range(batch_size):
    #             query_sim = torch.matmul(image_features[j], query_features[j].T).cpu().detach().unsqueeze(0)
    #             verb_sim = torch.matmul(image_features[j], verb_features[j].T).cpu().detach()
    #             noun_sim = torch.matmul(image_features[j], noun_features[j].T).cpu().detach()
    #             similarity_matrix = torch.cat((query_sim, verb_sim, noun_sim))
    #             all_preds.append(similarity_matrix)
    #     all_preds = torch.stack(all_preds)
    #     metrics = egohoi_accuracy_metrics(all_preds)
    #     print(metrics)
    #     return metrics


def validate_image(
    args, val_loader, hoi_detector, sam_model, prompt_type, use_half=False
):
    hoi_detector.eval()
    sam_model.eval()
    is_multi_obj = args.multiobj_track
    tgt_type = args.target_type
    input_dicts = initialize_inputs(no_cuda=args.no_cuda)

    for i, image in enumerate(val_loader):
        # hoi_detector pre-processing
        input_dicts, im_scales = prepare_inputs(image, **input_dicts)
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
        # sam pre-processing
        if prompt_type == "box":
            box_input = prepare_boxes(hand_dets, obj_dets, is_multi_obj, tgt_type)
            prompt_inputs = dict(box=box_input, multimask_output=args.multimask_output)
        else:
            raise NotImplementedError(f"{args.prompt_type} not implemented yet")
        # model inference
        masks, sam_scores = sam_prediction(sam_model, image, prompt_inputs)


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
        "--thresh-hoid-score",
        type=float,
        default=0.5,
        help="the threshold score that a detection of hoi detector is good",
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

    args = parser.parse_args()
    model_names = args.model_names

    # data preparation
    if args.dataset_name == "visor":
        val_dataset = VISOR()
        val_loader = nn.DataLoader(val_dataset, ...)
    elif args.dataset_name == "demo":
        image = Image.open("examples/ego4d_example.png")
        image = np.array(image.convert("RGB"))

    # model preparation
    hoi_detector, sam_model = get_model(args, model_names)

    # inference
    if args.dataset_name == "visor":
        validate_visor(args, val_loader, hoi_detector, sam_model)
    elif args.dataset_name == "demo":
        validate_image(args, [image], hoi_detector, sam_model, args.prompt_type)
