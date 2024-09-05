import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from hoid_model.utils.net_utils import vis_detections_filtered_objects_PIL
from tqdm import tqdm

from star_hoi.data.utils import (
    detection_post_process,
    initialize_inputs,
    prepare_boxes,
    prepare_hoid_inputs,
    prepare_sam_image_inputs,
)
from star_hoi.evaluation.utils import prepare_output_for_evaluator
from star_hoi.utils.visualize import show_masks


def sam_image_prediction(sam_model, image, prompt_input_list):
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
    hoid_input_dicts = initialize_inputs(no_cuda=args.no_cuda)
    # evaluator init
    evaluator.reset()

    for i, inputs in tqdm(enumerate(val_loader), total=len(val_loader)):
        # hoi_detector processing
        # from BGR, 750, 1333-->750, 1333, BGR
        im_hoi = inputs[0]["image"].permute(1, 2, 0).numpy()
        # from BGR to RGB, for the sake of sam2 input
        image = np.ascontiguousarray(im_hoi[..., ::-1])
        hoid_input_dicts, im_scales = prepare_hoid_inputs(
            im_hoi, hoid_test_short_size, args.hoid_test_max_size, **hoid_input_dicts
        )
        # hoid model inference
        (rois, cls_prob, bbox_pred, _, _, _, _, _, loss_list) = hoi_detector(
            **hoid_input_dicts
        )
        hand_dets, obj_dets = detection_post_process(
            args,
            rois,
            cls_prob,
            bbox_pred,
            loss_list,
            im_scales,
            hoid_input_dicts["im_info"],
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
        masks, sam_scores = sam_image_prediction(sam_model, image, prompt_list)
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
    hoid_test_short_size = (args.hoid_test_short_size,)
    hoid_input_dicts = initialize_inputs(no_cuda=args.no_cuda)
    # im_hoi = cv2.imread("examples/ego4d_example.png")
    for i, image in enumerate(val_loader):
        im_hoi = image[..., ::-1]
        # hoi_detector pre-processing
        hoid_input_dicts, im_scales = prepare_hoid_inputs(
            im_hoi, hoid_test_short_size, args.hoid_test_max_size, **hoid_input_dicts
        )  # input: BGR
        # model inference
        (rois, cls_prob, bbox_pred, _, _, _, _, _, loss_list) = hoi_detector(
            **hoid_input_dicts
        )
        hand_dets, obj_dets = detection_post_process(
            args,
            rois,
            cls_prob,
            bbox_pred,
            loss_list,
            im_scales,
            hoid_input_dicts["im_info"],
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
        # if hand_prompt_type == "box" and obj_prompt_type == "box":
        #     box_input = prepare_boxes(hand_dets, obj_dets, is_multi_obj, tgt_type)
        #     prompt_inputs = dict(box=box_input, multimask_output=args.multimask_output)
        # else:
        #     raise NotImplementedError(
        #         f"{hand_prompt_type} or {args.obj_prompt_type} error"
        #     )
        # # sam inference
        # masks, sam_scores = sam_image_prediction(sam_model, image, [prompt_inputs])
        box_input = prepare_boxes(hand_dets, obj_dets, is_multi_obj, "hoi")
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
        masks, sam_scores = sam_image_prediction(sam_model, image, prompt_list)
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
