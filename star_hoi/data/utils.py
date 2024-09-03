import contextlib
import io
import logging
import os
import pdb
import random
import time

import cv2
import numpy as np
import pycocotools.mask as mask_util
import torch
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode
from hoid_model.roi_layers import nms
from hoid_model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from hoid_model.utils.blob import im_list_to_blob

from star_hoi.utils.utils import calculate_center

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3

PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
TRAIN_BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)
TRAIN_BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
pascal_classes = np.asarray(["__background__", "targetobject", "hand"])

logger = logging.getLogger(__name__)


def _load_epick_json(
    json_file, image_root, dataset_name=None, extra_annotation_keys=None
):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection, instance segmentation,
    and person keypoints annotations.

    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        dataset_name (str or None): the name of the dataset (e.g., coco_2017_train).
            When provided, this function will also do the following:

            * Put "thing_classes" into the metadata associated with this dataset.
            * Map the category ids into a contiguous range (needed by standard dataset format),
              and add "thing_dataset_id_to_contiguous_id" to the metadata associated
              with this dataset.

            This option should usually be provided, unless users need to load
            the original json content and apply more processing manually.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format (See
        `Using Custom Datasets </tutorials/datasets.html>`_ ) when `dataset_name` is not None.
        If `dataset_name` is None, the returned `category_ids` may be
        incontiguous and may not conform to the Detectron2 standard format.

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from pycocotools.coco import COCO

    # json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        meta.thing_classes = thing_classes

        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning(
                    """
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
"""
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = coco_api.loadImgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'iscrowd': 0,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    if "minival" not in json_file:
        # The popular valminusminival & minival annotations for COCO2014 contain this bug.
        # However the ratio of buggy annotations there is tiny and does not affect accuracy.
        # Therefore we explicitly white-list them.
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(
            ann_ids
        ), "Annotation ids in '{}' are not unique!".format(json_file)

    imgs_anns = list(zip(imgs, anns))
    logger.info(
        "Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file)
    )

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"] + (
        extra_annotation_keys or []
    )
    # print(f'extra_annotation_keys = {extra_annotation_keys}')
    # extra_annotation_keys = ['handside', 'incontact', 'offset', 'object_id']
    # print(f'extra_annotation_keys = {extra_annotation_keys}')

    num_instances_without_valid_segmentation = 0

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            assert anno["image_id"] == image_id

            assert (
                anno.get("ignore", 0) == 0
            ), '"ignore" in COCO json file is not supported.'

            obj = {key: anno[key] for key in ann_keys if key in anno}
            if "bbox" in obj and len(obj["bbox"]) == 0:
                raise ValueError(
                    f"One annotation of image {image_id} contains empty 'bbox' value! "
                    "This json does not have valid COCO format."
                )

            segm = anno.get("segmentation", None)
            if segm:  # either list[list[float]] or dict(RLE)
                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                else:
                    # filter out invalid polygons (< 3 points)
                    segm = [
                        poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6
                    ]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

            keypts = anno.get("keypoints", None)
            if keypts:  # list[int]
                for idx, v in enumerate(keypts):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # Therefore we assume the coordinates are "pixel indices" and
                        # add 0.5 to convert to floating point coordinates.
                        keypts[idx] = v + 0.5
                obj["keypoints"] = keypts

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if id_map:
                annotation_category_id = obj["category_id"]
                try:
                    obj["category_id"] = id_map[annotation_category_id]
                except KeyError as e:
                    raise KeyError(
                        f"Encountered category_id={annotation_category_id} "
                        "but this id does not exist in 'categories' of the json file."
                    ) from e
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process.  Please "
            "check https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html carefully"
        )
    return dataset_dicts


def _get_image_blob(im, test_short_size=(800,), test_max_size=1200):
    """Converts an image into a network input.
    Arguments:
      im (ndarray): a color image in BGR order
    Returns:
      blob (ndarray): a data blob holding an image pyramid
      im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in test_short_size:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > test_max_size:
            im_scale = float(test_max_size) / float(im_size_max)
        im = cv2.resize(
            im_orig,
            None,
            None,
            fx=im_scale,
            fy=im_scale,
            interpolation=cv2.INTER_LINEAR,
        )
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


def prepare_hoid_inputs(
    image,
    test_short_size,
    test_max_size,
    im_data,
    im_info,
    gt_boxes,
    num_boxes,
    box_info,
):
    """
    prepare inputs to the hoi_detectors
    input image: (H, W, BGR)
    """
    im = image

    blobs, im_scales = _get_image_blob(im, test_short_size, test_max_size)
    assert len(im_scales) == 1, "Only single-image batch implemented"
    im_blob = blobs
    im_info_np = np.array(
        [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32
    )
    im_data_pt = torch.from_numpy(im_blob)
    im_data_pt = im_data_pt.permute(0, 3, 1, 2)
    im_info_pt = torch.from_numpy(im_info_np)

    with torch.no_grad():
        im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
        im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
        gt_boxes.resize_(1, 1, 5).zero_()
        num_boxes.resize_(1).zero_()
        box_info.resize_(1, 1, 5).zero_()

    return (
        dict(
            im_data=im_data,
            im_info=im_info,
            gt_boxes=gt_boxes,
            num_boxes=num_boxes,
            box_info=box_info,
        ),
        im_scales,
    )


def initialize_inputs(no_cuda):
    """initialize inputs to the hoi_detectors, and adaptive to image sizes"""
    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    box_info = torch.FloatTensor(1)

    # use cuda
    if no_cuda is False:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    return dict(
        im_data=im_data,
        im_info=im_info,
        gt_boxes=gt_boxes,
        num_boxes=num_boxes,
        box_info=box_info,
    )


def prepare_points(boxes, neg_boxes=None, point_type="center"):
    """
    Given box, select points as sam's point inputs, neg_boxes is to provide negative center points
    args:
        boxes: (N, 4): x1, y1, x2, y2
        point_type: 'center', 'box_coords'
    """
    if point_type == "center":
        pos_points = np.array([calculate_center(box) for box in boxes]).reshape(
            -1, 1, 2
        )  # (B1, 1, 2)
        B1 = pos_points.shape[0]
        pos_labels = np.ones(B1, dtype=int).reshape(B1, 1)
        if neg_boxes is not None:
            # inter hands with objects
            neg_points_inter = np.array([calculate_center(box) for box in neg_boxes])[
                np.newaxis, ...
            ].repeat(
                B1, axis=0
            )  # B1, B2, 2
            if B1 > 1:
                # intra-hands
                neg_points_intra = np.zeros((B1, B1 - 1, 2))  # B1, B1-1, 2
                for i in range(B1):
                    other_coords = np.delete(pos_points, i, axis=0)
                    neg_points_intra[i] = other_coords.reshape(-1, 2)
                neg_points = np.concatenate(
                    [neg_points_intra, neg_points_inter], axis=1
                )  # B1, B2+B1-1, 2
            else:
                neg_points = neg_points_inter
            neg_labels = np.zeros(neg_points.shape[1], dtype=int)[np.newaxis].repeat(
                B1, 0
            )  # B1, B1+B2 - 1

            labels = np.concatenate([pos_labels, neg_labels], axis=1)  # B1, 1+(B1+B2-1)
            points = np.concatenate(
                [pos_points, neg_points], axis=1
            )  # B1, 1+(B1+B2-1), 2
        else:
            labels = np.array([1] * (pos_points.shape[0]), np.int32).reshape(B1, 1)
            points = pos_points
    else:
        raise NotImplementedError(f"{point_type} not implemented yet")

    return points, labels


def prepare_boxes(hand_dets, obj_dets, is_multi_obj, tgt_type):
    """prompt type for sam: box
    tgt_type: in ['obj', 'hand', 'hoi']
    """
    box_input = None
    if obj_dets is not None:
        obj_boxes = obj_dets[:, :4]
    if hand_dets is not None:
        hand_boxes = hand_dets[:, :4]

    if tgt_type == "obj":
        if obj_dets is not None:
            box_input = obj_boxes
    elif tgt_type == "hand":
        if hand_dets is not None:
            box_input = hand_boxes
    elif tgt_type == "hoi":
        if obj_dets is None and hand_dets is None:
            return None
        elif obj_dets is not None and hand_dets is None:
            box_input = obj_boxes
        elif obj_dets is None and hand_dets is not None:
            box_input = hand_boxes
        else:
            box_input = np.concatenate([obj_boxes, hand_boxes], axis=0)  # (N, 4)
    else:
        raise NotImplementedError(f"{tgt_type} not implemented yet")

    return box_input


def detection_post_process(
    args, rois, cls_prob, bbox_pred, loss_list, im_scales, im_info, thresh_hoid_score
):
    """post process & nms for hoi detection results, the box score default: 0.5
    returns:
        hand_dets (ndarray): N, 10=4(box,[x1, y1, x2, y2], without normalize) + 1(cls_score,float)+1(contact,int,[0, 4])+3(offset_vector,float, relative distance of obj regarding hands, only hands make senses)+1(left/right hand,bool)
        obj_dets (ndarray): N, 10, the same format with hand_dets
    """
    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]  # 0, x00
    # extract predicted params
    contact_vector = loss_list[0][0]  # hand contact state info, [1. 300, 5]
    offset_vector = loss_list[1][
        0
    ].detach()  # offset vector (factored into a unit vector and a magnitude), [1, 300, 3]
    lr_vector = loss_list[2][0].detach()  # hand side info (left/right) # [1, 300, 1]

    # get hand contact
    _, contact_indices = torch.max(contact_vector, 2)
    contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()  # [300, 1]

    # get hand side
    lr = torch.sigmoid(lr_vector) > 0.5
    lr = lr.squeeze(0).float()  # 300, 1

    # Apply bounding-box regression deltas
    box_deltas = bbox_pred.data  # 1, 300, 12
    box_deltas = (
        box_deltas.view(-1, 4) * torch.FloatTensor(TRAIN_BBOX_NORMALIZE_STDS).cuda()
        + torch.FloatTensor(TRAIN_BBOX_NORMALIZE_MEANS).cuda()
    )
    box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))  # 1. 300, 12

    pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
    pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
    pred_boxes /= im_scales[0]

    scores = scores.squeeze()  # 300, 3
    pred_boxes = pred_boxes.squeeze()
    obj_dets, hand_dets = None, None
    for j in xrange(1, len(pascal_classes)):
        # inds = torch.nonzero(scores[:,j] > thresh).view(-1)
        if pascal_classes[j] == "hand":
            inds = torch.nonzero(scores[:, j] > thresh_hoid_score).view(-1)
        elif pascal_classes[j] == "targetobject":
            # j == 1
            inds = torch.nonzero(scores[:, j] > thresh_hoid_score).view(-1)  # (9, )

        # if there is det
        if inds.numel() > 0:
            cls_scores = scores[:, j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            if args.class_agnostic:
                cls_boxes = pred_boxes[inds, :]
            else:
                cls_boxes = pred_boxes[inds][
                    :, j * 4 : (j + 1) * 4
                ]  # pred_boxes: [300, 12], cls_boxes: [9, 4]

            cls_dets = torch.cat(
                (
                    cls_boxes,
                    cls_scores.unsqueeze(1),
                    contact_indices[inds],
                    offset_vector.squeeze(0)[inds],
                    lr[inds],
                ),
                1,
            )
            cls_dets = cls_dets[
                order
            ]  # 9, 10：4(box)+1(cls_score)+1(contact)+3(offset_vec)+1(lr)
            keep = nms(
                cls_boxes[order, :], cls_scores[order], args.hoid_test_nms
            )  # indices: [0, 2], 有两个物体
            cls_dets = cls_dets[keep.view(-1).long()]
            if pascal_classes[j] == "targetobject":
                obj_dets = cls_dets.cpu().numpy()
            if pascal_classes[j] == "hand":
                hand_dets = cls_dets.cpu().numpy()
    return hand_dets, obj_dets


def prepare_sam_image_inputs(
    args,
    hand_dets,
    obj_dets,
    is_multi_obj,
    multimask_output,
    hand_prompt_type,
    obj_prompt_type,
):
    """
    prepare inputs for sam
    """
    prompt_list = []
    if hand_prompt_type == "box" and obj_prompt_type == "box":
        box_input = prepare_boxes(hand_dets, obj_dets, is_multi_obj, "hoi")
        prompt_inputs = dict(box=box_input, multimask_output=multimask_output)
        prompt_list.append(prompt_inputs)
    elif hand_prompt_type == "point" and obj_prompt_type == "box":
        if obj_dets is not None:
            # object prompt prepare
            obj_box = prepare_boxes(hand_dets, obj_dets, is_multi_obj, tgt_type="obj")
            obj_prompt_inputs = dict(box=obj_box, multimask_output=multimask_output)
            prompt_list.append(obj_prompt_inputs)
        if hand_dets is not None:
            # hand prompt prepare
            hand_point, hand_point_labels = prepare_points(
                hand_dets[:, :4], obj_dets[:, :4] if obj_dets is not None else None
            )
            hand_prompt_inputs = dict(
                point_coords=hand_point,
                point_labels=hand_point_labels,
                multimask_output=multimask_output,
            )
            prompt_list.append(hand_prompt_inputs)
    else:
        raise NotImplementedError(
            f"{hand_prompt_type} or {args.obj_prompt_type} not Implemented yet, \
            currently (hand, obj) only supports: (box, box), (point, box) inputs"
        )
    if prompt_list == []:
        prompt_list.append(dict(multimask_output=multimask_output))
    return prompt_list


def concate_hoi(hand_dets, obj_dets, idx):
    """
    concatenate objects with hands along idx, e.g., handsides, contacts
    """
    if obj_dets is None and hand_dets is None:
        # random added
        rand_num = np.zeros([1, 10])
        return rand_num[:, idx]
    elif obj_dets is not None and hand_dets is None:
        output = obj_dets[:, idx]
    elif obj_dets is None and hand_dets is not None:
        output = hand_dets[:, idx]
    else:
        output = np.concatenate([obj_dets[:, idx], hand_dets[:, idx]], axis=0)

    return output
