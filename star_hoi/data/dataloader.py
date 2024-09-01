from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.utils.data as torchdata
from detectron2.data.common import MapDataset
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.transforms.augmentation_impl import Resize, ResizeShortestEdge


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def build_detection_test_loader(
    dataset: Union[List[Any], torchdata.Dataset],
    sampler: Optional[torchdata.Sampler] = None,
    batch_size: int = 1,
    num_workers: int = 0,
    collate_fn: Optional[Callable[[List[Any]], Any]] = None,
):

    # augmentations=[
    #     ResizeShortestEdge(
    #         short_edge_length=(800, 800), max_size=1333, sample_style="choice"
    #     )
    # ],
    input_to_mapper = dict(
        is_train=False,
        augmentations=[Resize((1080, 1920))],
        image_format="BGR",
        use_instance_mask="True",
        instance_mask_format="bitmask",
    )
    mapper = DatasetMapper(**input_to_mapper)
    dataset = MapDataset(dataset, mapper)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
    )


if __name__ == "__main__":
    extra_keys = ["handside", "isincontact", "offset", "incontact_object_bbox"]
    version = "/data2/xbs/data/VISOR/epick_visor_coco_hos"
    json_file = f"{version}/annotations/val.json"
    image_root = f"{version}/val"
    from dataset import build_dataset

    # dataset_dicts = _load_epick_json(json_file, image_root, None, extra_annotation_keys=extra_keys)
    dataset = build_dataset(None, "visor_image", json_file, image_root)
    from sampler import InferenceSampler

    sampler = InferenceSampler(len(dataset))
    dataloader = build_detection_test_loader(dataset, sampler)
    for i, batch in enumerate(dataloader):
        # image size: 3, 750, 1333, height和width记录的是原来image的长和宽， 1920，1080
        print(i, batch)
        import pdb

        pdb.set_trace()
