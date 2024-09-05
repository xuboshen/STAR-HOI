import numpy as np
import torch
from detectron2.structures import Boxes

from star_hoi.data.instances import Instances
from star_hoi.data.utils import concate_hoi
from star_hoi.utils.utils import numpy_to_torch_dtype


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
