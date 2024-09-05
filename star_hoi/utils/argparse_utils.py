import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--foo", help="foo help")
    # common arguments
    parser.add_argument(
        "--image-path",
        type=str,
        default="images/ego4d_example.png",
        help="which is also the image_root",
    )
    parser.add_argument(
        "--anno-path",
        type=str,
        default="VISOR/epick_visor_coco_hos/annotations/val.json",
        help="the annotation file",
    )
    parser.add_argument(
        "--vis",
        help="default is False, test the results",
        action="store_true",
    )
    parser.add_argument(
        "--vis-freq",
        type=int,
        default=100,
        help="the visualization save frequency",
    )
    parser.add_argument(
        "--debug",
        help="debug model, default is false",
        action="store_true",
    )
    parser.add_argument(
        "--dataset-name",
        default="demo_image",
        type=str,
        choices=["demo_image", "demo_video", "visor_image", "ego4d_video"],
        help="choose from visor_sparse, visor_dense, demo_image, and demo_video",
    )
    parser.add_argument(
        "--print-freq",
        default=100,
        type=int,
        help="print frequency during inference",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Default is False, i.e., with cuda",
    )
    parser.add_argument(
        "--test-sam2",
        nargs="*",
        help="support more than one argument and merge into a list.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="outputs/demo",
        help="save to output path",
    )
    parser.add_argument(
        "--vis-path",
        type=str,
        default="image_vis",
        help="visualization path to save",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="numbers of data processes init in dataloader",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Default is False, i.e., do not save",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="results",
        help="default save to ./results",
    )
    parser.add_argument(
        "--multigpu-inference",
        action="store_true",
        help="whether to use multiple gpus for inference, default is not",
    )

    # hoid model parameters
    parser.add_argument(
        "--hoid-test-max-size",
        type=int,
        default=1200,
        help="the largest width of input image size to hoi detector model",
    )
    parser.add_argument(
        "--hoid-test-short-size",
        type=int,
        default=800,
        help="the shortest length of input image size to hoi detector model",
    )
    parser.add_argument(
        "--thresh-hoid-score",
        type=float,
        default=0.3,
        help="the threshold score that a detection of hoi detector is good",
    )
    parser.add_argument(
        "--target-type",
        type=str,
        default="hoi",
        choices=["obj", "lh", "rh", "hoi"],
        help="choose tracking targets after hoi detection",
    )
    parser.add_argument(
        "--hoid-model",
        type=str,
        default="res101",
        choices=["vgg16", "res50", "res101", "res152"],
        help="types of fasterRCNN",
    )
    parser.add_argument(
        "--hoid-test-nms",
        type=float,
        default=0.3,
        help="test nms score",
    )
    parser.add_argument(
        "--hoid-checkpoint",
        type=str,
        default="checkpoints/faster_rcnn_1_8_132028.pth",
        help="path to the hoid checkpoint",
    )
    parser.add_argument(
        "--cag",
        dest="class_agnostic",
        help="whether perform class_agnostic bbox regression, default is False",
        action="store_true",
    )

    # sam2 model image parameters
    parser.add_argument(
        "--thresh-sam-score",
        type=float,
        default=0,
        help="the threshold score that a object is good",
    )
    parser.add_argument(
        "--multiobj-track",
        action="store_true",
        help="Default to be False, i.e. default is single_object",
    )
    parser.add_argument(
        "--hand-prompt-type",
        default="box",
        choices=["box", "point", "mask", "mixed"],
        help="prompt for initial frame, the hand prompt type for sam2.",
    )
    parser.add_argument(
        "--obj-prompt-type",
        type=str,
        default="box",
        choices=["box", "point", "mask", "mixed"],
        help="prompt for initial frame, the object prompt type for sam2 input",
    )
    parser.add_argument(
        "--sam-model-cfg",
        type=str,
        default="sam2_hiera_l.yaml",
        choices=[
            "sam2_hiera_t.yaml",
            "sam2_hiera_s.yaml",
            "sam2_hiera_b+.yaml",
            "sam2_hiera_l.yaml",
        ],
        help="model config of sam model",
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=str,
        default="checkpoints/sam2_hiera_large.pt",
        help="path to the sam checkpoint",
    )
    parser.add_argument(
        "--multimask-output",
        action="store_true",
        help="Default is False, i.e., only one mask output, which is suitable for non-ambiguous prompts (e.g., click + box)",
    )

    # sam2 video configurations
    parser.add_argument(
        "--clip-length",
        type=int,
        default=12,
        help="number of frames that we tracks",
    )

    # Downstream datasets
    parser.add_argument(
        "--eval-task",
        type=str,
        default="hand_obj",
        choices=["hand_obj"],
        help="Used for eval_task of single-image VISOR-HOS Benchmark",
    )

    return parser.parse_args()
