import torch, torchvision
from torchviz import make_dot
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.config import get_cfg
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances

import numpy as np
import cv2
from PIL import Image
import os
import sys
import random
import argparse

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)



from vos.detection.modeling.regnet import build_regnet_fpn_backbone

# create a parser object
parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--backbone', type=str, help='Description of the backbone argument')
parser.add_argument('--id', type=str, help='Description of the in-distribution dataset argument')
args = parser.parse_args()
backbone = args.backbone
id = args.id
dataset_name = id + "-train"
# %%
dataset = fo.load_dataset(dataset_name)
print("Dataset '%s' loaded with %d samples" % (dataset_name, len(dataset)))
VOC_THING_CLASSES = ['person',
                     'bird',
                     'cat',
                     'cow',
                     'dog',
                     'horse',
                     'sheep',
                     'airplane',
                     'bicycle',
                     'boat',
                     'bus',
                     'car',
                     'motorcycle',
                     'train',
                     'bottle',
                     'chair',
                     'dining table',
                     'potted plant',
                     'couch',
                     'tv',
                     ]

BDD_THING_CLASSES = ['pedestrian',
                    'rider',
                    'car',
                    'truck',
                    'bus',
                    'train',
                    'motorcycle',
                    'bicycle',
                    'traffic light',
                    'traffic sign']
label_list = VOC_THING_CLASSES if id == 'voc' else BDD_THING_CLASSES
label_dict = {i:label for i, label in enumerate(label_list)}
# %%
dataset.shuffle(seed=42)

# Take the first 10,000 samples after shuffling
subset = dataset.take(20000) if len(dataset) > 20000 else dataset

# %%
cfg = get_cfg()
cfg.merge_from_file(f"vos/detection/configs/BDD-Detection/faster-rcnn/{backbone}.yaml")
cfg.MODEL.WEIGHTS = f"model_final_vos_{backbone}_{id}.pth" 
cfg.MODEL.DEVICE='cuda'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(label_list)

# %%
model = build_model(cfg)
model.eval()
checkpointer = DetectionCheckpointer(model)
checkpointer.load(cfg.MODEL.WEIGHTS)

aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def inference(inputs):
    with torch.no_grad():
        images = model.preprocess_image(inputs)  
        features = model.backbone(images.tensor)  
        proposals, _ = model.proposal_generator(images, features, None)  # RPN

        features_ = [features[f] for f in model.roi_heads.box_in_features]
        box_features = model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])
        box_features = model.roi_heads.box_head(box_features)  # features of all 1k candidates
        predictions = model.roi_heads.box_predictor(box_features)
        pred_instances, pred_inds = model.roi_heads.box_predictor.inference(predictions, proposals)
        pred_instances = model.roi_heads.forward_with_given_boxes(features, pred_instances)

        # output boxes, masks, scores, etc
        pred_instances = model._postprocess(pred_instances, inputs, images.image_sizes)  # scale box to orig size
        # features of the proposed boxes
        feats = box_features[pred_inds]
    return pred_instances, feats

i = 0
feats_list = []
with fo.ProgressBar() as pb:
    for sample in pb(subset):
        image = cv2.imread(sample.filepath)
        height, width = image.shape[:2]
        image = aug.get_transform(image).apply_image(image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)).to(device)
        inputs = [{"image": image, "height": height, "width": width}]

        preds, feats = inference(inputs)
        feats = feats.cpu().detach().numpy()
        boxes = preds[0]["instances"].pred_boxes.tensor.cpu().detach().numpy()
        classes = preds[0]["instances"].pred_classes.cpu().detach().numpy()
        scores = preds[0]["instances"].scores.cpu().detach().numpy()

        detections = []
        

        for score, label, box, feat in zip(scores, classes, boxes, feats):
            x1, y1, x2, y2 = box
            rel_box = [x1/width, y1/height, (x2 - x1) / width, (y2 - y1) / height]
            label = label_dict[label]
            detections.append(
                fo.Detection(
                    label=label,
                    bounding_box=rel_box,
                    confidence=score,
                    feature_idx=i
                ),
            )
            i += 1
            feats_list.append(feat)
        sample["prediction"] = fo.Detections(detections=detections)
        sample.save()
print("Finished adding predictions")
feats_npy = np.array(feats_list)
print(f"Saved {feats_npy.shape[0]} features")
matched_view = subset.filter_labels("prediction", F("confidence") > 0.7, only_matches=True)
results = matched_view.evaluate_detections(
    "prediction",
    gt_field="detections",
    eval_key="eval",
    compute_mAP=True)
results.print_report()
tp_prediction_view = matched_view.filter_labels("prediction", F("eval") == "tp")
class_names = list(label_dict.values())
features_idx_dict = {cls:[] for cls in class_names}
with fo.ProgressBar() as pb:
    for sample in pb(tp_prediction_view):
        for detection in sample.prediction.detections:
            label_pred = detection.label
            feature_idx = detection.feature_idx
            features_idx_dict[label_pred].append(feature_idx)
for label in class_names:
    idx = features_idx_dict[label]
    filename = f"feats_{label}.npy"
    np.save(filename, feats_npy[idx], allow_pickle=True)
    print(f"Saved {len(idx)} features for {label}")
