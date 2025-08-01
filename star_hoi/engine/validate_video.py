import json
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
    prepare_sam_frame_inputs,
)
from star_hoi.utils.utils import mask_decode, mask_encode
from star_hoi.utils.visualize import show_mask


def select_best_hoi_frame(
    args,
    frames,
    sam_video_model,
    hoi_detector,
    hand_prompt_type,
    obj_prompt_type,
    is_multi_obj,
    vis_results=True,
    iter_number=0,
):
    """
    choose the best hoi frame masks for sam2 tracking initialization
    how to define the "best": detects the max number of boxes + one of the hands is contacting object (occlusion, may not be the best),
    接触瞬间的那一帧最重要：有接触、少遮挡
    args:
        hand_dets (ndarray):
            N, 10=4(box,[x1, y1, x2, y2],:4) + 1(cls_score,float, 4)+1(contact,int,[0, 4], 5)+3(offset_vector,float, 6:9)+1(left/right hand,bool, 9)
            contact:
                0 N: no contact
                1 S: self contact
                2 O: other person contact
                3 P: portable object contact
                4 F: stationary object contact (e.g.furniture)
        frames (ndarray, uint8): B, H, W, 3,
        obj_dets (ndarray): N, 10, the same format with hand_dets
    returns:
        hoi_boxes (nd.array): we will also save the numbers into hdf5 files
        best_frame_idx (int): the best frame to select
        prompt_inputs (List[dicts]): prompt_inputs to the sam2
    """
    # initializations
    init_frame = False
    hoid_test_short_size = (args.hoid_test_short_size,)
    hoid_input_dicts = initialize_inputs(no_cuda=args.no_cuda)
    hoi_boxes = np.zeros((12, 4, 10), dtype=float)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = sam_video_model.init_state(video_frames=frames)
    # processing
    for idx, frame in enumerate(frames):
        # H, W, 3
        im_hoi = frame[..., ::-1]
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
        # prepare saving box outputs
        if hand_dets is not None:
            hoi_boxes[idx, : hand_dets.shape[0], :] = hand_dets
        if obj_dets is not None:
            hoi_boxes[idx, 2 : (2 + obj_dets.shape[0]), :] = obj_dets
        if vis_results is True:
            # raw image
            plt.figure(figsize=(10, 10))
            plt.imshow(frame)
            plt.axis("off")
            save_path = f"{args.vis_path}/{iter_number}"
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(f"{save_path}/frame_{idx}.png", dpi=200)
            plt.close()

            # hoid output
            im2show = np.copy(im_hoi)
            im2show = vis_detections_filtered_objects_PIL(
                im2show,
                obj_dets,
                hand_dets,
                args.thresh_hoid_score,
                args.thresh_hoid_score,
                font_path="hand_object_detector/lib/hoid_model/utils/times_b.ttf",
            )
            im2show.save(f"{save_path}/frame_{idx}_det.png")
        # filter no contacts
        if hand_dets is None or all(hand_dets[:, 5] == 0) is True or init_frame is True:
            continue
        print(f"The initial frame is {idx}")
        init_frame = True
        # sam inference
        masks, obj_ids, obj_id_mapping = sam_frame_prediction(
            args,
            state,
            sam_video_model,
            frames,
            idx,
            hand_dets,
            obj_dets,
            is_multi_obj,
            hand_prompt_type,
            obj_prompt_type,
        )
        # fix box_input if exists None
        box_input = prepare_boxes(hand_dets, obj_dets, is_multi_obj, "hoi")
        if box_input is None:
            box_input = np.zeros([1, 4])
            masks = np.zeros_like(masks, dtype=masks.dtype)
        if vis_results is True:
            # sam2 output
            plt.figure(figsize=(10, 10))
            plt.imshow(frame)
            for i, out_obj_id in enumerate(obj_ids):
                show_mask(masks[i], plt.gca(), obj_id=out_obj_id, borders=False)
            plt.axis("off")
            plt.savefig(
                f"{save_path}/frame_{idx}_init_mask.png",
                dpi=200,
            )
            plt.close()

    return hoi_boxes, masks, state, obj_id_mapping


def sam_video_prediction(
    args, predictor, state, frames, vis_results=True, iter_number=0
):
    """
    sam prediction: input images, output region masks
    args:
        prompt_list: [obj, hand]
    returns:
        masks: N, 1, H, W or 1, H, W. obj_masks->hand_masks
        scores: N, 1 or N
    """
    video_segments = {}

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # video regular sequence: [init_frames, init_frames++...]
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            state
        ):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            state, reverse=True
        ):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        # video reverse seqeunce: [init_frames, init_frames--...]
        if vis_results is True:
            # render the segmentation results every few frames
            vis_frame_stride = 1
            plt.close("all")
            for out_frame_idx in range(0, len(frames), vis_frame_stride):
                plt.figure(figsize=(6, 4))
                plt.title(f"frame {out_frame_idx}")
                plt.imshow(frames[out_frame_idx])
                for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                    show_mask(out_mask, plt.gca(), obj_id=out_obj_id, borders=False)
                save_path = f"{args.vis_path}/{iter_number}"
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(
                    f"{save_path}/frame_{out_frame_idx}_propogate_mask.png",
                    dpi=200,
                )
                plt.close()

    return video_segments


def sam_frame_prediction(
    args,
    state,
    predictor,
    frames,
    idx,
    hand_dets,
    obj_dets,
    is_multi_obj,
    hand_prompt_type,
    obj_prompt_type,
):
    """
    sam prediction: input frame, output region masks
    args:
        prompt_list: [obj, hand]
    returns:
        masks: N, 1, H, W or 1, H, W. obj_masks->hand_masks
        no scores returned
    """
    mask_list = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.reset_state(state)
        # sam image processing
        prompt_list, obj_id_mapping = prepare_sam_frame_inputs(
            args,
            state,
            idx,
            hand_dets,
            obj_dets,
            is_multi_obj,
            hand_prompt_type,
            obj_prompt_type,
        )
        if prompt_list == [{}]:
            return np.zeros(1, frames.shape[2], frames.shape[3]), None, None
        for prompt_input in prompt_list:
            frame_idx, out_obj_ids, masks = predictor.add_new_points_or_box(
                **prompt_input
            )
        mask_list.extend([(msk > 0.0).detach().cpu().numpy() for msk in masks])
    _, H, W = mask_list[0].shape
    if len(mask_list) > 1:
        masks = np.stack(mask_list)
    else:
        masks = np.array(mask_list).reshape(1, H, W)

    return masks, out_obj_ids, obj_id_mapping


@torch.no_grad()
def validate_demo_video(
    args,
    val_loader,
    hoi_detector,
    sam_video_model,
    hand_prompt_type,
    obj_prompt_type,
    use_half=False,
):
    """
    only used for demo, i.e. without evaluation
    here, val_loader = [video], video = N frames
    """
    print("demo image processing begins...")
    hoi_detector.eval()
    is_multi_obj = args.multiobj_track

    for idx, frames in enumerate(val_loader):
        hoi_boxes, masks, state, obj_id_mapping = select_best_hoi_frame(
            args,
            frames,
            sam_video_model,
            hoi_detector,
            hand_prompt_type,
            obj_prompt_type,
            is_multi_obj,
        )
        video_segments = sam_video_prediction(args, sam_video_model, state, frames)
        if args.vis and idx % args.vis_freq == 0:
            pass

        def save_boxes(hoi_boxes):
            pass

        def save_masks(video_segments):
            pass

        save_boxes(hoi_boxes)
        save_masks(video_segments)


@torch.no_grad()
def validate_ego4d_video(
    args,
    val_loader,
    hoi_detector,
    sam_video_model,
    hand_prompt_type,
    obj_prompt_type,
    use_half=False,
):
    """
    used for data preprocessing without evaluation
    """
    print("Inference Begins...")
    hoi_detector.eval()
    is_multi_obj = args.multiobj_track
    for i, inputs in tqdm(enumerate(val_loader), total=len(val_loader)):
        frames, video_uid, narration_time = (
            inputs[0]["frame"],
            inputs[0]["uid"],
            inputs[0]["timestamp"],
        )
        if args.save_results and os.path.exists(
            os.path.join(args.save_path, video_uid, narration_time + ".json")
        ):
            continue
        if frames is None or frames.sum() == 0:
            continue
        hoi_boxes, masks, state, obj_id_mapping = select_best_hoi_frame(
            args,
            np.array(frames.detach().cpu(), dtype=np.uint8),
            sam_video_model,
            hoi_detector,
            hand_prompt_type,
            obj_prompt_type,
            is_multi_obj,
            vis_results=(args.vis is True and i % args.vis_freq == 0),
            iter_number=i,
        )
        video_segments = sam_video_prediction(
            args,
            sam_video_model,
            state,
            np.array(frames, dtype=np.uint8),
            vis_results=(args.vis is True and i % args.vis_freq == 0),
            iter_number=i,
        )
        if args.save_results:

            def save_masks(
                video_uid: str,
                narration_time: str,
                video_segments,
                obj_id_mapping,
            ):
                # 创建mask字典
                mask_dicts = {}
                for frame_idx, mask_list_dicts in video_segments.items():
                    mask_dicts[frame_idx] = {}
                    for obj_id, mask in mask_list_dicts.items():
                        mask = mask[0]
                        mask_dicts[frame_idx][obj_id] = {
                            "type": obj_id_mapping[obj_id],
                            "mask": mask_encode(mask),
                            "H": mask.shape[0],
                            "W": mask.shape[1],
                        }

                save_dir = os.path.join(args.save_path, video_uid)
                os.makedirs(save_dir, exist_ok=True)

                merged_file_path = os.path.join(save_dir, "merged_file.json")

                if os.path.exists(merged_file_path):
                    with open(merged_file_path, "r") as f:
                        merged_data = json.load(f)
                else:
                    merged_data = {}

                merged_data[narration_time] = mask_dicts

                with open(merged_file_path, "w") as f:
                    json.dump(merged_data, f, indent=4)

            save_masks(video_uid, narration_time, video_segments, obj_id_mapping)

            def save_hoi(box):
                save_dir = os.path.join(args.save_path, video_uid)
                os.makedirs(save_dir, exist_ok=True)

                merged_file_box_path = os.path.join(save_dir, "merged_file_box.json")

                if os.path.exists(merged_file_box_path):
                    with open(merged_file_box_path, "r") as f:
                        merged_data = json.load(f)
                else:
                    merged_data = {}

                merged_data[narration_time] = box.tolist()

                with open(merged_file_box_path, "w") as f:
                    json.dump(merged_data, f, indent=4)

            # 调用函数
            save_hoi(hoi_boxes)
