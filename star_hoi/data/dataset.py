import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .utils import _load_epick_json


class VISOR_Image(Dataset):
    """
    {'file_name': '/data2/xbs/data/VISOR/epick_visor_coco_hos/val/P01_107_frame_0000000155.jpg', 'height': 1080, 'width': 1920, 'image_id': 0, 'annotations': [{'iscrowd': 0, 'bbox': [1330, 728, 176, 352], 'category_id': 1, 'handside': 1, 'isincontact': 1, 'offset': [0.11480196111539398, -0.993388398223001, 0.4268219769412067], 'segmentation': [[1436.28, 740.73, 1434.79, 733.65, 1429.02, 729.46, 1416.96, 728.63, 1409.66, 733.77, 1405.7, 743.26, 1402.33, 751.83, 1392.2, 750.79, 1387.22, 752.34, 1382.7, 757.37, 1376.19, 758.66, 1374.99, 755.67, 1371.11, 753.88, 1366.03, 753.58, 1358.86, 756.57, 1354.97, 761.35, 1354.67, 771.81, 1384.25, 839.04, 1383.06, 846.21, 1377.68, 843.22, 1367.82, 825.59, 1351.98, 787.05, 1345.11, 781.07, 1340.33, 780.47, 1333.76, 782.56, 1331.37, 785.25, 1330.77, 796.01, 1344.46, 846.73, 1349.76, 862.58, 1353.03, 867.15, 1352.77, 874.28, 1346.27, 883.33, 1343.79, 892.38, 1345.49, 911.46, 1348.52, 924.5, 1348.51, 941.85, 1346.71, 970.67, 1344.79, 989.31, 1339.07, 1032.77, 1332.58, 1078.1, 1333.87, 1080.0, 1471.1, 1080.0, 1477.12, 1079.69, 1476.36, 1054.55, 1474.62, 1033.65, 1468.35, 995.03, 1462.25, 951.66, 1461.08, 930.89, 1465.12, 910.85, 1475.39, 893.48, 1494.57, 871.17, 1506.53, 852.3, 1506.47, 840.53, 1504.24, 831.35, 1498.22, 821.23, 1492.76, 816.98, 1472.06, 813.15, 1463.18, 810.11, 1461.02, 806.1, 1468.55, 793.9, 1470.1, 788.44, 1469.32, 784.03, 1456.08, 768.19, 1448.29, 766.11, 1439.83, 767.02]], 'bbox_mode': <BoxMode.XYWH_ABS: 1>}, {'iscrowd': 0, 'bbox': [1015, 0, 905, 960], 'category_id': 2, 'handside': -1, 'isincontact': -1, 'offset': [-1, -1, -1], 'segmentation': [[1919.93, 0.69, 1920.0, 785.01, 1579.89, 860.58, 1577.98, 872.01, 1575.44, 875.82, 1567.19, 877.09, 1559.57, 875.82, 1556.39, 866.93, 1482.56, 885.14, 1494.57, 871.17, 1506.53, 852.3, 1506.47, 840.53, 1504.24, 831.35, 1498.22, 821.23, 1492.76, 816.98, 1472.06, 813.15, 1463.18, 810.11, 1461.02, 806.1, 1468.55, 793.9, 1470.1, 788.44, 1469.32, 784.03, 1456.08, 768.19, 1448.29, 766.11, 1439.83, 767.02, 1436.28, 740.73, 1434.79, 733.65, 1429.02, 729.46, 1416.96, 728.63, 1409.66, 733.77, 1405.7, 743.26, 1402.33, 751.83, 1392.2, 750.79, 1387.22, 752.34, 1382.7, 757.37, 1376.19, 758.66, 1374.99, 755.67, 1371.11, 753.88, 1366.03, 753.58, 1358.86, 756.57, 1354.97, 761.35, 1354.67, 771.81, 1384.25, 839.04, 1383.06, 846.21, 1377.68, 843.22, 1367.82, 825.59, 1351.98, 787.05, 1345.11, 781.07, 1340.33, 780.47, 1333.76, 782.56, 1331.37, 785.25, 1330.77, 796.01, 1344.46, 846.73, 1349.76, 862.58, 1353.03, 867.15, 1352.77, 874.28, 1346.27, 883.33, 1343.79, 892.38, 1345.49, 911.46, 1347.14, 918.54, 1175.37, 960.91, 1169.65, 960.91, 1158.86, 960.28, 1015.34, 468.12, 1118.76, 0.52]], 'bbox_mode': <BoxMode.XYWH_ABS: 1>}]}
    """

    def __init__(self, anno_path, image_root):
        super().__init__()
        extra_keys = ["handside", "isincontact", "offset", "incontact_object_bbox"]
        self.file_list = _load_epick_json(
            anno_path, image_root, None, extra_annotation_keys=extra_keys
        )

    def load_image(self, image_path):
        return Image.open(image_path)

    def __getitem__(self, idx):
        file_dict = self.file_list[idx]
        # image = load_image(file_dict['file_name'])
        # anno_list = file_dict['annotations']
        return file_dict  # image, mask, box

    def __len__(self):
        return len(self.file_list)


def build_dataset(args, dataset_name, anno_path, image_root):
    if dataset_name == "visor_image":
        dataset = VISOR_Image(anno_path, image_root)
    return dataset


if __name__ == "__main__":
    extra_keys = ["handside", "isincontact", "offset", "incontact_object_bbox"]
    version = "/data2/xbs/data/VISOR/epick_visor_coco_hos"
    json_file = f"{version}/annotations/val.json"
    image_root = f"{version}/val"
    # dataset_dicts = _load_epick_json(json_file, image_root, None, extra_annotation_keys=extra_keys)
    dataset = build_dataset(None, "visor_image", json_file, image_root)
