# %%
import torch
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
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data import DatasetMapper, build_detection_test_loader
from detectron2.data import get_detection_dataset_dicts

import numpy as np
import cv2
from PIL import Image
import os
import argparse
from pickle import load

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from vos.detection.modeling.regnet import build_regnet_fpn_backbone
# create a parser object
parser = argparse.ArgumentParser(description='Description of your program')

# add an optional "--model" argument 
parser.add_argument('--backbone', type=str, help='Description of the backbone argument')
parser.add_argument('--id', type=str, help='Description of the in-distribution dataset argument')
parser.add_argument('--tau', nargs='+', type=float, help='Description of the tau argument')

# %%
args = parser.parse_args()

id = args.id
backbone = args.backbone
taus = args.tau

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

cfg = get_cfg()
cfg.merge_from_file(f"vos/detection/configs/BDD-Detection/faster-rcnn/{backbone}.yaml")
cfg.MODEL.WEIGHTS = f"model_final_vos_{backbone}_{id}.pth" 
cfg.MODEL.DEVICE='cuda'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(label_list)
model = build_model(cfg)
model.eval()
checkpointer = DetectionCheckpointer(model)
checkpointer.load(cfg.MODEL.WEIGHTS)

aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
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

# %%
delta = 0.1
for tau in taus:
    print(f"tau: {tau}")
    monitors_dict = {}
    for class_name in label_list:
        monitor_path = f"monitors/{id}/{backbone}/{class_name}/monitor_for_clustering_parameter" + "_tau_" + str(tau) + ".pkl"
        # monitor_path = f"Monitors/{class_name}/monitor_for_clustering_parameter" + "_tau_" + str(tau) + ".pkl"
        if os.path.exists(monitor_path):
            with open(monitor_path, 'rb') as f:
                monitor = load(f)
            for i in range(len(monitor.good_ref)):
                monitor.good_ref[i].ivals = monitor.good_ref[i].ivals*np.array([1-delta, 1+delta])
            monitors_dict[class_name] = monitor
        else:
            print(f"monitor for {class_name} not found")
    
    eval_list = ["bdd-val", "ID-bdd-OOD-coco", "OOD-open"] if id == "bdd" else ["voc-val", "ID-voc-OOD-coco", "OOD-open"]
    for dataset_name in eval_list:
        print(f"evaluating on {dataset_name}")
        dataset_val = fo.load_dataset(dataset_name)
        i = 0
        feats_list = []
        with fo.ProgressBar() as pb:
            for sample in pb(dataset_val):
                image = cv2.imread(sample.filepath)
                height, width = image.shape[:2]
                image = aug.get_transform(image).apply_image(image)
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)).to(device)
                inputs = [{"image": image, "height": height, "width": width}]
                preds, feats = inference(inputs)
                feats = feats.cpu().detach().numpy()
                boxes = preds[0]["instances"].get("pred_boxes").tensor.cpu().detach().numpy()
                scores = preds[0]["instances"].get("scores").cpu().detach().numpy()
                classes = preds[0]["instances"].pred_classes.cpu().detach().numpy()
                
                detections = []
                oods = []             
                for label, score, box, feat in zip(classes, scores, boxes, feats):
                    label = label_dict[label]
                    x1, y1, x2, y2 = box
                    rel_box = [x1/width, y1/height, (x2 - x1) / width, (y2 - y1) / height]
                    verdict = monitors_dict[label].make_verdicts(feat[np.newaxis, :])[0]
                    detections.append(
                        fo.Detection(
                            label=label,
                            bounding_box=rel_box,
                            confidence=score,
                            verdict=verdict
                            # feature_idx=i
                        ),
                    )
                    # i += 1
                    if not verdict:
                        oods.append(
                            fo.Detection(
                                label="OOD",
                                bounding_box=rel_box
                            ),
                        )
                sample["OOD"] = fo.Detections(detections=oods)
                sample["prediction"] = fo.Detections(detections=detections)
                sample.save()
        if dataset_name == "bdd-val" or dataset_name == "voc-val":
            gt_field = "detections"
            results = dataset_val.evaluate_detections(
            "prediction",
            gt_field=gt_field,
            eval_key="eval",
            compute_mAP=True,
            )
            results.print_report()
            eval_patches = dataset_val.to_evaluation_patches("eval")
            confusion_matrix = eval_patches.count_values("type")
            print(f"Confusion Matrix: {confusion_matrix}\n")
            verdict_true_dict = eval_patches.filter_labels("prediction",(F("verdict") == True)).count_values("type")
            verdict_false_dict = eval_patches.filter_labels("prediction",(F("verdict") == False)).count_values("type")
            print(f"When verdict is Accept: {verdict_true_dict}\n")
            print(f"When verdict is Reject: {verdict_false_dict}\n")
            print(f"FPR: {verdict_true_dict['fp'] / (verdict_false_dict['fp'] + verdict_true_dict['fp'])}\n")
            print(f"TPR: {verdict_true_dict['tp'] / (verdict_false_dict['tp'] + verdict_true_dict['tp'])}\n")
        else:
            dataset_val = dataset_val.filter_labels("prediction", F("confidence") > 0.5, only_matches=True)
            verdict_true_dict = len(dataset_val.filter_labels("prediction",(F("verdict") == True)))
            verdict_false_dict = len(dataset_val.filter_labels("prediction",(F("verdict") == False)))
            print(f"When verdict is Accept: {verdict_true_dict}\n")
            print(f"When verdict is Reject: {verdict_false_dict}\n")
            print(f"FPR: {verdict_true_dict / (verdict_true_dict + verdict_false_dict)}\n")
        
    
 


