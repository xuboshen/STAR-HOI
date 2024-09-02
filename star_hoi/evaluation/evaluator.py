"""
This is the evaluation code for epick.
Difference from coco evaluation: evaluate handside and contact state.
"""
import os
import pdb

import numpy as np
import pycocotools.mask as mask_util
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import COCOEvaluator
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.structures import Boxes, BoxMode, pairwise_iou

from ..data.utils import _load_epick_json
from .hos_postprocessing import combineHO_postprocessing, hos_postprocessing


def register_epick_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    # extra label for each object
    extra_keys = ["handside", "isincontact", "offset", "incontact_object_bbox"]
    DatasetCatalog.register(
        name,
        lambda: _load_epick_json(
            json_file, image_root, name, extra_annotation_keys=extra_keys
        ),
    )

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )


class EPICKEvaluator(COCOEvaluator):
    def __init__(self, dataset_name, output_dir=None, eval_task=None, tasks=None):
        super().__init__(dataset_name, output_dir=output_dir)
        self.eval_task = eval_task
        assert self.eval_task in [
            "hand_obj",
            "handside",
            "contact",
            "combineHO",
        ], "Error: target not in ['hand_obj', 'handside', 'contact', 'combineHO']"
        print(f"**Evaluation target: {self.eval_task}")
        if tasks is not None:
            self._tasks = tasks
        self._metadata = MetadataCatalog.get(dataset_name)
        print(f"dataset name = {dataset_name}")
        print(f"meta data = {self._metadata}")

    def process(self, inputs, outputs):
        """
        Re-format the inputs and outputs to make handside and contact predictions as another 4 classes.

        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            # pdb.set_trace()
            # print(f'input = {input}')
            # print(f'output = {output}\n')
            if "instances" in output:
                # tmp = output
                instances = output["instances"].to(self._cpu_device)
                if self.eval_task == "hand_obj":
                    # post-processing: link hand and obj
                    output["instances"] = hos_postprocessing(instances)
                elif self.eval_task in ["handside", "contact"]:
                    # only keep hand preds
                    output["instances"] = instances[instances.pred_classes == 0]
                elif self.eval_task == "combineHO":
                    # combine hand and obj mask
                    output["instances"] = combineHO_postprocessing(instances)

                prediction["instances"] = instances_to_coco_json_handside_or_contact(
                    output["instances"], input["image_id"], eval_task=self.eval_task
                )

            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            if len(prediction) > 1:
                self._predictions.append(prediction)
            # print(f"Process: out={len(tmp['instances'])}; #hand={len(tmp['instances'][tmp['instances'].pred_classes==0]) if len(tmp['instances'])!=0 else ''}; #obj={len(tmp['instances'][tmp['instances'].pred_classes==1]) if len(tmp['instances'])!=0 else ''}; ")
            # print(f"Process: out2={len(output['instances'])}; #hand={len(output['instances'][output['instances'].pred_classes==0]) if len(output['instances'])!=0 else ''};  #obj={len(output['instances'][output['instances'].pred_classes==1]) if len(output['instances'])!=0 else ''}\n")
            # import pdb; pdb.set_trace()


def instances_to_coco_json_handside_or_contact(instances, img_id, eval_task=None):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.
    fixerror: for VISOR-HOS, the hand id is 1, the object id is 2, instead of 0, 1 respectively.

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    # print(f'instances = {len(instances)}')
    num_instance = len(instances)
    if num_instance == 0:
        return []

    assert eval_task in [
        "hand_obj",
        "handside",
        "contact",
        "combineHO",
    ], "Error: evaluation target should be either 'hand_obj', 'handside', 'contact', 'combineHO'"

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    if eval_task in ["hand_obj", "combineHO"]:
        scores = instances.scores.tolist()
        classes = instances.pred_classes.tolist()
    else:
        if eval_task == "handside":
            preds = instances.pred_handsides.numpy()
        elif eval_task == "contact":
            preds = instances.pred_contacts.numpy()
        scores = np.max(preds, axis=1).tolist()
        classes = np.argmax(preds, axis=1).tolist()

    has_mask = instances.has("pred_masks")
    if has_mask:
        # use RLE to encode the masks, because they are too large and takes memory
        # since this evaluator stores outputs of the entire dataset
        rles = [
            mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            for mask in instances.pred_masks
        ]
        for rle in rles:
            # "counts" is an array encoded by mask_util as a byte-stream. Python3's
            # json writer which always produces strings cannot serialize a bytestream
            # unless you decode it. Thankfully, utf-8 works out (which is also what
            # the pycocotools/_mask.pyx does).
            rle["counts"] = rle["counts"].decode("utf-8")

    has_keypoints = instances.has("pred_keypoints")
    if has_keypoints:
        keypoints = instances.pred_keypoints

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k] + 1,
            "bbox": boxes[k],
            "score": scores[k],
        }
        if has_mask:
            result["segmentation"] = rles[k]
        if has_keypoints:
            # In COCO annotations,
            # keypoints coordinates are pixel indices.
            # However our predictions are floating point coordinates.
            # Therefore we subtract 0.5 to be consistent with the annotation format.
            # This is the inverse of data loading logic in `datasets/coco.py`.
            keypoints[k][:, :2] -= 0.5
            result["keypoints"] = keypoints[k].flatten().tolist()
        results.append(result)
    return results


def build_evaluator(args, eval_task, output_path):
    register_epick_instances(
        "epick_visor_2022_val_hos", {}, f"{args.anno_path}", f"{args.image_path}"
    )
    MetadataCatalog.get("epick_visor_2022_val_hos").thing_classes = ["hand", "object"]
    evaluator = (
        EPICKEvaluator(
            "epick_visor_2022_val_hos",
            output_dir=args.output_path,
            eval_task="hand_obj",
            tasks=["segm"],
        ),
    )
    return evaluator[0]
