import numpy as np
import torch


def numpy_to_torch_dtype(numpy_dtype):
    """
    Maps numpy dtype to the corresponding torch dtype.
    """
    dtype_mapping = {
        "np.float32": torch.float32,
        "np.float64": torch.float64,
        "np.int32": torch.int32,
        "np.int64": torch.int64,
        "np.uint8": torch.uint8,
        "np.int8": torch.int8,
        "np.uint16": torch.uint16,
        "np.int16": torch.int16,
        "np.bool_": torch.bool,
        "bool": torch.bool
        # Add more mappings as needed
    }
    return dtype_mapping.get(
        str(numpy_dtype), torch.float32
    )  # Default to float32 if dtype is unknown


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
