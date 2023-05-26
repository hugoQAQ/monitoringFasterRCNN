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
import pickle
import gradio as gr

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from vos.detection.modeling.regnet import build_regnet_fpn_backbone
# # create a parser object
# parser = argparse.ArgumentParser(description='Description of your program')

# # add an optional "--model" argument 
# parser.add_argument('--backbone', type=str, help='Description of the backbone argument')
# parser.add_argument('--id', type=str, help='Description of the in-distribution dataset argument')

# # %%
# args = parser.parse_args()
# id = args.id
# backbone = args.backbone

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
# VOC_THING_CLASSES = ["airplane",
#  "bicycle",
#  "bird",
#  "boat",
#  "bottle",
#  "bus",
#  "car",
#  "cat",
#  "chair",
#  "cow",
# "dining table",
#  "dog",
#  "horse",
#  "motorcycle",
#  "person",
#  "potted plant",
#  "sheep",
#  "couch",
#  "train",
#  "tv"]

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

def extract(id, backbone, dataset_name, progress=gr.Progress()):
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

    if dataset_name == "voc-ood":
        dataset = fo.load_dataset("voc-val")
        dataset_val = dataset.load_saved_view("ood")
    else:
        dataset_val = fo.load_dataset(dataset_name)
    i = 0
    feats_list = []
    for sample in progress.tqdm(dataset_val, "Extracting features"):
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
        for label, score, box, feat in zip(classes, scores, boxes, feats):
            label = label_dict[label]
            x1, y1, x2, y2 = box
            rel_box = [x1/width, y1/height, (x2 - x1) / width, (y2 - y1) / height]
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
    if dataset_name == "bdd-val" or dataset_name == "voc-val":
        gt_field = "detections"
        results = dataset_val.evaluate_detections(
        "prediction",
        gt_field=gt_field,
        eval_key="eval",
        compute_mAP=True,
        )
        report = results.print_report()
        feats_npy = np.array(feats_list)
        tp_prediction_view = dataset_val.filter_labels("prediction", F("eval") == "tp")
        features_idx_dict = {cls:[] for cls in label_list}
        for sample in progress.tqdm(tp_prediction_view, "Saving TP features"):
            for detection in sample.prediction.detections:
                label_pred = detection.label
                feature_idx = detection.feature_idx
                features_idx_dict[label_pred].append(feature_idx)
        feats_tp_dict = {cls:feats_npy[features_idx_dict[cls]] for cls in label_list}
        file_path = f'eval_feats/{id}/{backbone}/{dataset_name}_feats_tp_dict.pickle'
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
                pickle.dump(feats_tp_dict, f)

        fp_prediction_view = dataset_val.filter_labels("prediction", F("eval") == "fp")
        features_idx_dict = {cls:[] for cls in label_list}
        for sample in progress.tqdm(fp_prediction_view, "Saving FP features"):
            for detection in sample.prediction.detections:
                label_pred = detection.label
                feature_idx = detection.feature_idx
                features_idx_dict[label_pred].append(feature_idx)
        feats_fp_dict = {cls:feats_npy[features_idx_dict[cls]] for cls in label_list}
        file_path = f'eval_feats/{id}/{backbone}/{dataset_name}_feats_fp_dict.pickle'
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
                pickle.dump(feats_fp_dict, f)
    else:
        feats_npy = np.array(feats_list)
        dataset_val = dataset_val.filter_labels("prediction", F("confidence") > 0.5, only_matches=True)
        features_idx_dict = {cls:[] for cls in label_list}
        for sample in progress.tqdm(dataset_val, "Saving FP features"):
            for detection in sample.prediction.detections:
                label_pred = detection.label
                feature_idx = detection.feature_idx
                features_idx_dict[label_pred].append(feature_idx)
        feats_fp_dict = {cls:feats_npy[features_idx_dict[cls]] for cls in label_list}
        feats_fp_dict = {cls:feats_npy[features_idx_dict[cls]] for cls in label_list}
        file_path = f'eval_feats/{id}/{backbone}/{dataset_name}_feats_fp_dict.pickle'
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
                pickle.dump(feats_fp_dict, f) 
        report = "None"  
    return report

dataset_list = {"bdd": ["ID-bdd-OOD-coco", "OOD-open", "bdd-val", "voc-ood"], "voc": ["ID-voc-OOD-coco", "OOD-open", "voc-val"]}
with gr.Blocks() as demo:
    def get_name(id):
        return gr.update(choices=dataset_list[id], value=dataset_list[id][0])
    id = gr.Radio(["voc", "bdd"], label="select in-distribution dataset")
    backbone = gr.Radio(["resnet", "regnet"], label="select backbone")
    dataset_name = gr.Dropdown([], label="select out-dtribution dataset")
    btn = gr.Button("Extract")
    id.change(get_name, inputs=id, outputs=dataset_name)
    output = gr.Textbox(label="output", max_lines=100)
    btn.click(extract, inputs=[id, backbone, dataset_name], outputs=output)
demo.queue().launch()





