import base64
import json
import os
import sys

import cv2
import numpy as np
import pycocotools.mask as mask_util
import torch


def resize_with_aspect_ratio(image, target_size):
    h, w = image.shape[:2]

    # 计算缩放比例，目标是最短边等于 target_size
    if h < w:
        scale = target_size / h
    else:
        scale = target_size / w

    # 计算新尺寸
    new_w = int(w * scale)
    new_h = int(h * scale)

    # 调整大小
    resized_img = cv2.resize(image, (new_w, new_h))
    return resized_img


def mask_encode(mask):
    """
    encode masks -> binary -> json saved, default size: 1280 * 720
    """
    binary_encoded = mask_util.encode(np.array(mask, order="F", dtype="uint8"))
    encoded_mask = base64.b64encode(binary_encoded["counts"]).decode("utf-8")
    return encoded_mask


def mask_decode():
    pass


def calculate_center(bb):
    """
    calculate center for box (4,) with (x1, y1, x2, y2)
    """
    return [(bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2]


class Tee:
    """
    to redirect output to files
    """

    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for file in self.files:
            file.write(data)
            file.flush()  # Ensure it gets written immediately

    def flush(self):
        for file in self.files:
            file.flush()


def redirect_output(output_file):
    """
    Redirects terminal output (stdout and stderr) to both the terminal and a file.

    Parameters:
    output_file (str): Path to the file where the output will be saved.
    """
    # Open the file in write mode
    log_file = open(output_file, "w")

    # Create a Tee object that writes to both the terminal (sys.__stdout__) and the log file
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = sys.stdout


def save_args(args):
    """
    Save the arguments from argparse to a file named 'args.log' in the specified output directory.

    Parameters:
    args (Namespace): The argparse.Namespace object containing the parsed arguments.
    """
    # Ensure the output directory exists
    os.makedirs(args.output_path, exist_ok=True)
    # Define the path to the log file
    log_file_path = os.path.join(args.output_path, "args.log")
    # Convert the Namespace object to a dictionary
    args_dict = vars(args)
    # Save the arguments to the log file in a human-readable JSON format
    with open(log_file_path, "w") as log_file:
        json.dump(args_dict, log_file, indent=4)

    print(f"Arguments have been saved to {log_file_path}")


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
