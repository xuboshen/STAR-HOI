import time

import cv2
import numpy as np
import torch
from hoi_model.utils.blob import im_list_to_blob
from hoid_model.roi_layers import nms
from hoid_model.rpn.bbox_transform import bbox_transform_inv, clip_boxes

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3

PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
TEST_MAX_SIZE = 1000
TRAIN_BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)
TRAIN_BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
pascal_classes = np.asarray(["__background__", "targetobject", "hand"])


def _get_image_blob(im, scales=(600,)):
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

    for target_size in scales:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > TEST_MAX_SIZE:
            im_scale = float(TEST_MAX_SIZE) / float(im_size_max)
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


def prepare_inputs(image, im_data, im_info, gt_boxes, num_boxes, box_info):
    """prepare inputs to the hoi_detectors"""
    im = image

    blobs, im_scales = _get_image_blob(im)
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


def prepare_boxes(hand_dets, obj_dets, is_multi_obj, tgt_type):
    """prompt type for sam: box
    tgt_type: in ['obj', 'lh', 'rh', 'hoi']
    """
    box_input = None
    obj_boxes = obj_dets[:, :4]
    hand_boxes = hand_dets[:, :4]
    if tgt_type == "obj":
        box_input = obj_boxes
    elif tgt_type == "lh":
        pass
    elif tgt_type == "rh":
        pass
    elif tgt_type == "hoi":
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
