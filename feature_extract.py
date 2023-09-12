import torch
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.config import get_cfg
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model


import numpy as np
import cv2
import os
import argparse
import tqdm
import time
import h5py
import pickle

import fiftyone as fo
from fiftyone import ViewField as F
torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
KITTI_THING_CLASSES = ["Car", "Pedestrian", "Cyclist", "Van", "Truck", "Tram"]
SPEED_THING_CLASSES = ['100kph','120kph','20kph','30kph','40kph','5kph','50kph','60kph','70kph','80kph']
NU_THING_CLASSES = ['car','truck','trailer','bus','construction_vehicle','bicycle','motorcycle','pedestrian','traffic_cone','barrier']

def inference(model, inputs):
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
        feats = box_features[pred_inds].cpu().numpy()   
    return pred_instances, feats

def save_features(feats_npy, dataset_view, file_path):
    features_idx_dict = {cls:[] for cls in label_list}
    for sample in tqdm.tqdm(dataset_view, desc="Saving features"):
        for detection in sample.prediction.detections:
            label_pred = detection.label
            feature_idx = detection.feature_idx
            features_idx_dict[label_pred].append(feature_idx)
    feats_dict = {cls:feats_npy[features_idx_dict[cls]] for cls in label_list}
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
            pickle.dump(feats_dict, f)

def extract(dataset, id, backbone):
    aug = T.AugmentationList([T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST),
        ]
    )
    dataset_name = dataset.name
    i = 0
    feats_list = []
    for sample in tqdm.tqdm(dataset, desc="Extracting features"):
        image = cv2.imread(sample.filepath)
        height, width = image.shape[:2]
        input = T.AugInput(image)
        transform = aug(input)
        image = input.image
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)).to(device)
        inputs = [{"image": image, "height": height, "width": width}]

        preds, feats = inference(model, inputs)
        boxes = preds[0]["instances"].pred_boxes.tensor.cpu().detach().numpy()
        classes = preds[0]["instances"].pred_classes.cpu().detach().numpy()
        scores = preds[0]["instances"].scores.cpu().detach().numpy()
        
        feats_list.extend(feats)
        if i == 1000:
            np.save('feats.npy', feats_list)
        
        detections = []
        for score, label, box in zip(scores, classes, boxes):
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
        sample["prediction"] = fo.Detections(detections=detections)
        sample.save()
    feats_npy = np.array(feats_list)
    if dataset.name.endswith("train"):
        with h5py.File(f'feats_{id}-train_{backbone}.h5', 'w') as f:
            dset = f.create_dataset(f"feats_{id}-train_{backbone}", data=feats_npy)
    if id == "kitti":
        with h5py.File(f"feats_{id}-{dataset_name}_{backbone}.h5", 'w') as f:
            dset = f.create_dataset(f"feats_{id}-{dataset_name}_{backbone}", data=feats_npy)
        print(f"Saved {feats_npy.shape[0]} features")
    if dataset.name.endswith("val") or dataset.name.endswith("train"):
        results = dataset.evaluate_detections(
            "prediction",
            gt_field="detections",
            eval_key="eval",
            compute_mAP=True)
        results.print_report()
        print("mAP: ", results.mAP())
        if dataset.name.endswith("train"):
            tp_prediction_view = dataset.filter_labels("prediction", F("eval") == "tp")
            save_features(feats_npy, tp_prediction_view, f"train_feats/{id}/{backbone}/{dataset_name}_feats_tp_dict.pickle")
        elif dataset.name.endswith("val"):    
            tp_prediction_view = dataset.filter_labels("prediction", F("eval") == "tp")
            fp_prediction_view = dataset.filter_labels("prediction", F("eval") == "fp")
            save_features(feats_npy, tp_prediction_view, f"val_feats/{id}/{backbone}/{dataset_name}_feats_tp_dict.pickle")    
            save_features(feats_npy, fp_prediction_view, f"val_feats/{id}/{backbone}/{dataset_name}_feats_fp_dict.pickle")
    else:   
        save_features(feats_npy, dataset, f"val_feats/{id}/{backbone}/{dataset_name}_feats_fp_dict.pickle")  
        
    return f"Extraction for {dataset_name} is Done!"        


parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--id', type=str, help='Description of the in-distribution dataset argument')
parser.add_argument('--backbone', type=str)
args = parser.parse_args()
if args.id == 'voc':      
    label_list = VOC_THING_CLASSES
elif args.id == 'bdd':
    label_list = BDD_THING_CLASSES
elif args.id == 'kitti':
    label_list = KITTI_THING_CLASSES
elif args.id == 'speed' or args.id == 'prescan':
    label_list = SPEED_THING_CLASSES
elif args.id == 'nu':
    label_list = NU_THING_CLASSES

label_dict = {i:label for i, label in enumerate(label_list)}

cfg = get_cfg()
cfg.merge_from_file("resnet.yaml")
cfg.MODEL.WEIGHTS = f"model_final_{args.backbone}_{args.id}.pth" 
cfg.MODEL.DEVICE='cuda'
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(label_list)
model = build_model(cfg)
model.eval()
checkpointer = DetectionCheckpointer(model)
checkpointer.load(cfg.MODEL.WEIGHTS)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# extraction
print("Starting training set extraction")
t0 = time.time()
dataset_train = fo.load_dataset(f"{args.id}-train")
extract(dataset_train, args.id, args.backbone)
print(f"Extraction took {time.time() - t0} seconds")

print("Starting validation set extraction")
t0 = time.time()
dataset_list = {
    "bdd": ["ID-bdd-OOD-coco", "OOD-open", "bdd-val", "voc-ood"], 
    "voc": ["ID-voc-OOD-coco", "OOD-open", "voc-val"],
    "kitti": ["ID-bdd-OOD-coco", "OOD-open", "kitti-val", "voc-ood"],
    "speed": ["ID-bdd-OOD-coco", "OOD-open", "speed-val", "voc-ood"],
    "prescan": ["ID-bdd-OOD-coco", "OOD-open", "prescan-val", "voc-ood"],
    "nu": ["ID-bdd-OOD-coco", "OOD-open", "nu-val", "voc-ood"]}
for dataset_name in dataset_list[args.id]:
    print(f"Extracting {dataset_name}")
    dataset_val = fo.load_dataset(dataset_name)
    extract(dataset_val, args.id, args.backbone)
print(f"Validation set extraction took {time.time() - t0} seconds")