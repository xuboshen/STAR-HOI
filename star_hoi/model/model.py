import numpy as np
import torch
from hoid_model.faster_rcnn.resnet import resnet
from hoid_model.faster_rcnn.vgg16 import vgg16
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def build_hoid_model(args):
    # load model
    load_name = args.hoid_checkpoint
    pascal_classes = np.asarray(["__background__", "targetobject", "hand"])
    # initilize the network here.
    if args.hoid_model == "vgg16":
        fasterRCNN = vgg16(
            pascal_classes, pretrained=False, class_agnostic=args.class_agnostic
        )
    elif args.hoid_model == "res101":
        fasterRCNN = resnet(
            pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic
        )
    elif args.hoid_model == "res50":
        fasterRCNN = resnet(
            pascal_classes, 50, pretrained=False, class_agnostic=args.class_agnostic
        )
    elif args.hoid_model == "res152":
        fasterRCNN = resnet(
            pascal_classes, 152, pretrained=False, class_agnostic=args.class_agnostic
        )
    else:
        print("network is not defined")
        raise NotImplementedError(f"{args.hoid_model} not implemented yet.")

    fasterRCNN.create_architecture()

    print("load checkpoint %s" % (load_name))
    if args.no_cuda is False:
        checkpoint = torch.load(load_name)
    else:
        checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
    fasterRCNN.load_state_dict(checkpoint["model"])
    print("load hoi detector model successfully!")
    if args.no_cuda is False:
        fasterRCNN = fasterRCNN.cuda()

    return fasterRCNN


def build_model(args, model_names):
    for model_name in model_names:
        if model_name == "hoid":
            hoi_detector = build_hoid_model(args)
        elif model_name == "sam":
            sam_model = SAM2ImagePredictor(
                build_sam2(args.sam_model_cfg, args.sam_checkpoint)
            )
        else:
            raise NotImplementedError(f"{model_name} Not implemented yet")

    return hoi_detector, sam_model
