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

from vos.detection.modeling.regnet import build_regnet_fpn_backbone
from util import *

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
SPEED_THING_CLASSES = ['100km/h','120km/h','20km/h','30km/h','40km/h','5km/h','50km/h','60km/h','70km/h','80km/h']
NU_THING_CLASSES = ['car','truck','trailer','bus','construction_vehicle','bicycle','motorcycle','pedestrian','traffic_cone','barrier']
benchmark_vos = {"voc": {"resnet":[47.53, 51.33], "regnet":[47.77, 48.33]}, 
                "bdd": {"resnet":[44.27, 35.54, 49.85], "regnet":[36.61, 27.24, 41.05]},
                "kitti": {"resnet":[44.27, 35.54], "regnet":[36.61, 27.24]},
                "speed": {"resnet":["N/A", "N/A"], "regnet":["N/A", "N/A"]},
                "nu": {"resnet":["N/A", "N/A"], "regnet":["N/A", "N/A"]},
                }

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
        print(f"Saved {feats_npy.shape[0]} features")
    if dataset.name.endswith("val") or dataset.name.endswith("train"):
        results = dataset.evaluate_detections(
            "prediction",
            gt_field="detections",
            eval_key="eval",
            compute_mAP=True)
        # results.print_report()
        # print("mAP: ", results.mAP())
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

def extract_aug(dataset, id, backbone, epoch):
    dataset_name = dataset.name
    i = 0
    feats_list = []
    aug = T.AugmentationList([T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST),
        T.RandomBrightness(0.8, 1.8),
        T.RandomContrast(0.6, 1.3),
        T.RandomSaturation(0.8, 1.4),
        T.RandomLighting(0.8)
        ]
    )
    for e in range(epoch):
        print(f"Epoch {e+40}")
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
            print(f"Saved {feats_npy.shape[0]} features")
    
        results = dataset.evaluate_detections(
            "prediction",
            gt_field="detections",
            eval_key="eval",
            compute_mAP=True)
        tp_prediction_view = dataset.filter_labels("prediction", F("eval") == "tp")
        save_features(feats_npy, tp_prediction_view, f"train_feats/{id}/{backbone}/{dataset_name}_epoch{e+40}_feats_tp_dict.pickle")

def construct(id, backbone, taus):
    with open(f"train_feats/{id}/{backbone}/{id}-train_feats_tp_dict.pickle", 'rb') as f:
        feats_tp_dict = pickle.load(f)
    for label in tqdm.tqdm(label_list, desc="Contructing monitors"):
        feats_npy = feats_tp_dict[label]
        if feats_npy.shape[0] == 0:
            continue
        clustering_results  = features_clustering(feats_npy, taus)
        monitor_saving_folder = f"monitors/{id}/{backbone}/{label}/"
        monitor_construction_from_features(feats_npy, taus, clustering_results, label, monitor_saving_folder)
    return "Construction Done"

def load_monitors(id, backbone, label_list, tau):
    monitors_dict = dict()
    for class_name in label_list:
        monitor_path = f"monitors/{id}/{backbone}/{class_name}/monitor_for_clustering_parameter" + "_tau_" + str(tau) + ".pkl"
        if os.path.exists(monitor_path):
            with open(monitor_path, 'rb') as f:
                monitor = pickle.load(f)
            monitors_dict[class_name] = monitor
        else:
            print(f"monitor for {monitor_path} not found")
    return monitors_dict

def evaluate(id, backbone, tau):
    # benchmark
    benchmark = benchmark_vos[id][backbone]
    dataset_name = f"{id}-val"
    with open(f'val_feats/{id}/{backbone}/{dataset_name}_feats_tp_dict.pickle', 'rb') as f:
        feats_tp_dict = pickle.load(f)
    with open(f'val_feats/{id}/{backbone}/{dataset_name}_feats_fp_dict.pickle', 'rb') as f:
        feats_fp_dict = pickle.load(f)
    monitors_dict = load_monitors(id, backbone, label_list, tau)
    # make verdicts on ID data
    data_tp = []
    data_fp = []
    accept_sum = {"tp": 0, "fp": 0}
    reject_sum = {"tp": 0, "fp": 0}
    for label in tqdm.tqdm(label_list, desc="Evaluation on ID data"): 
        if label in monitors_dict:    
            verdict = monitors_dict[label].make_verdicts(feats_tp_dict[label])
            data_tp.append([label, len(verdict), np.sum(verdict)/len(verdict)])
            accept_sum["tp"] += np.sum(verdict)
            reject_sum["tp"] += len(verdict) - np.sum(verdict)   
            verdict = monitors_dict[label].make_verdicts(feats_fp_dict[label])
            data_fp.append([label, len(verdict), (len(verdict)-np.sum(verdict))/len(verdict)])
            accept_sum["fp"] += np.sum(verdict)
            reject_sum["fp"] += len(verdict) - np.sum(verdict)
    TPR = round((accept_sum['tp'] / (reject_sum['tp'] + accept_sum['tp'])*100), 2)
    FPR =  round((accept_sum['fp'] / (reject_sum['fp'] + accept_sum['fp'])*100), 2)
    if id == "voc":
        dataset_name = "PASCAL-VOC"
        eval_list = ["ID-voc-OOD-coco", "OOD-open"]
    elif id == "bdd": 
        id == "BDD100k"
        eval_list = ["ID-bdd-OOD-coco", "OOD-open", "voc-ood"]
    elif id == "kitti":
        id == "KITTI"
        eval_list = ["ID-bdd-OOD-coco", "OOD-open", "voc-ood"]
    elif id == "speed":
        id == "SPEED"
        eval_list = ["ID-bdd-OOD-coco", "OOD-open", "voc-ood"]
    elif id == "nu":
        id == "NU"
        eval_list = ["ID-bdd-OOD-coco", "OOD-open", "voc-ood"]
    df_summary = pd.DataFrame([[dataset_name, f"{TPR}%", "95%", f"{FPR}%"]], columns=["Dataset", "TPR", "TPR(benchmark)", "FPR"])
    
    data_ood = []
    i = 0
    for dataset_name in tqdm.tqdm(eval_list, desc="Evaluation on OOD data"):
        accept_sum = {"tp": 0, "fp": 0}
        reject_sum = {"tp": 0, "fp": 0}
        with open(f'val_feats/{id}/{backbone}/{dataset_name}_feats_fp_dict.pickle', 'rb') as f:
            feats_fp_dict = pickle.load(f)
        for label in label_list:
            if label in monitors_dict:
                verdict = monitors_dict[label].make_verdicts(feats_fp_dict[label])
                accept_sum["fp"] += np.sum(verdict)
                reject_sum["fp"] += len(verdict) - np.sum(verdict)
        FPR =  round((accept_sum['fp'] / (reject_sum['fp'] + accept_sum['fp'])*100), 2)
        data_ood.append([dataset_name, str(FPR)+"%", str(benchmark[i])+"%" if i<len(benchmark) else "N/A" ])
        i += 1
    
    # prepare dataframes
    df_ood = pd.DataFrame(data_ood, columns=["Dataset", "FPR", "FPR(benchmark)"])
    df_ood["Dataset"] = ["COCO", "Open Images"] if id == "voc" else ["COCO", "Open Images", "VOC-OOD"]
    print(df_summary)
    print(df_ood)

def enlarge_evaluation(id, backbone, tau, delta):
    # benchmark
    benchmark = benchmark_vos[id][backbone]
    dataset_name = f"{id}-val"
    with open(f'val_feats/{id}/{backbone}/{dataset_name}_feats_tp_dict.pickle', 'rb') as f:
        feats_tp_dict = pickle.load(f)
    with open(f'val_feats/{id}/{backbone}/{dataset_name}_feats_fp_dict.pickle', 'rb') as f:
        feats_fp_dict = pickle.load(f)
    monitors_dict = load_monitors(id, backbone, label_list, tau)
    for label in label_list:
        for i in range(len(monitors_dict[label].good_ref)):
            monitors_dict[label].good_ref[i].ivals = monitors_dict[label].good_ref[i].ivals*np.array([1-delta, 1+delta])
    # make verdicts on ID data
    data_tp = []
    data_fp = []
    accept_sum = {"tp": 0, "fp": 0}
    reject_sum = {"tp": 0, "fp": 0}

    for label in tqdm.tqdm(label_list, desc="Evaluation on ID data"):     
        if label in monitors_dict:    
            verdict = monitors_dict[label].make_verdicts(feats_tp_dict[label])
            data_tp.append([label, len(verdict), np.sum(verdict)/len(verdict)])
            accept_sum["tp"] += np.sum(verdict)
            reject_sum["tp"] += len(verdict) - np.sum(verdict)   
            verdict = monitors_dict[label].make_verdicts(feats_fp_dict[label])
            data_fp.append([label, len(verdict), (len(verdict)-np.sum(verdict))/len(verdict)])
            accept_sum["fp"] += np.sum(verdict)
            reject_sum["fp"] += len(verdict) - np.sum(verdict)
    TPR = round((accept_sum['tp'] / (reject_sum['tp'] + accept_sum['tp'])*100), 2)
    FPR =  round((accept_sum['fp'] / (reject_sum['fp'] + accept_sum['fp'])*100), 2)
    if id == "voc":
        dataset_name = "PASCAL-VOC"
        eval_list = ["ID-voc-OOD-coco", "OOD-open"]
    elif id == "bdd": 
        id == "BDD100k"
        eval_list = ["ID-bdd-OOD-coco", "OOD-open", "voc-ood"]
    elif id == "kitti":
        id == "KITTI"
        eval_list = ["ID-bdd-OOD-coco", "OOD-open", "voc-ood"]
    elif id == "speed":
        id == "SPEED"
        eval_list = ["ID-bdd-OOD-coco", "OOD-open", "voc-ood"]
    elif id == "nu":
        id == "NU"
        eval_list = ["ID-bdd-OOD-coco", "OOD-open", "voc-ood"]

    df_summary = pd.DataFrame([[dataset_name, f"{TPR}%", "95%", f"{FPR}%"]], columns=["Dataset", "TPR", "TPR(benchmark)", "FPR"])
    
    data_ood = []
    i = 0
    for dataset_name in tqdm.tqdm(eval_list, desc="Evaluation on OOD data"):
        accept_sum = {"tp": 0, "fp": 0}
        reject_sum = {"tp": 0, "fp": 0}
        with open(f'val_feats/{id}/{backbone}/{dataset_name}_feats_fp_dict.pickle', 'rb') as f:
            feats_fp_dict = pickle.load(f)
        for label in label_list:
            if label in monitors_dict: 
                verdict = monitors_dict[label].make_verdicts(feats_fp_dict[label])
                accept_sum["fp"] += np.sum(verdict)
                reject_sum["fp"] += len(verdict) - np.sum(verdict)
        FPR =  round((accept_sum['fp'] / (reject_sum['fp'] + accept_sum['fp'])*100), 2)
        data_ood.append([dataset_name, str(FPR)+"%", str(benchmark[i])+"%" if i<len(benchmark) else "N/A" ])
        i += 1
    
    # prepare dataframes
    df_ood = pd.DataFrame(data_ood, columns=["Dataset", "FPR", "FPR(benchmark)"])
    df_ood["Dataset"] = ["COCO", "Open Images"] if id == "voc" else ["COCO", "Open Images", "VOC-OOD"]
    print(df_summary)
    print(df_ood)

def tune_threshold(dataset_train, feats_npy, id, backbone, threshold, taus):
    dataset_filtered = dataset_train.filter_labels("prediction", F("confidence")>threshold)
    tp_prediction_view = dataset_filtered.filter_labels("prediction", F("eval") == "tp")
    save_features(feats_npy, tp_prediction_view, f"train_feats/{id}/{backbone}/{id}-train_feats_tp_dict.pickle")
    construct(id, backbone, taus)

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
elif args.id == 'speed':
    label_list = SPEED_THING_CLASSES
elif args.id == 'nu':
    label_list = NU_THING_CLASSES

label_dict = {i:label for i, label in enumerate(label_list)}

cfg = get_cfg()
cfg.merge_from_file(f"vos/detection/configs/{args.id.upper()}-Detection/faster-rcnn/vanilla_{args.backbone}.yaml")
cfg.MODEL.WEIGHTS = f"model_final_{args.backbone}_{args.id}.pth" 
cfg.MODEL.DEVICE='cuda'
model = build_model(cfg)
model.eval()
checkpointer = DetectionCheckpointer(model)
checkpointer.load(cfg.MODEL.WEIGHTS)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
aug = T.AugmentationList([T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST),
        # T.RandomBrightness(0.8, 1.8),
        # T.RandomContrast(0.6, 1.3),
        # T.RandomSaturation(0.8, 1.4),
        # T.RandomLighting(0.8)
        ]
    )
taus = [1.0, 0.5, 0.1, 0.05, 0.01]

# extraction
print("Starting training set extraction")
t0 = time.time()
dataset_train = fo.load_dataset(f"{args.id}-train")
extract(dataset_train, args.id, args.backbone)
# extract_aug(dataset_train, args.id, args.backbone, 10)
print(f"Extraction took {time.time() - t0} seconds")

print("Starting validation set extraction")
t0 = time.time()
dataset_list = {
    "bdd": ["ID-bdd-OOD-coco", "OOD-open", "bdd-val", "voc-ood"], 
    "voc": ["ID-voc-OOD-coco", "OOD-open", "voc-val"],
    "kitti": ["ID-bdd-OOD-coco", "OOD-open", "kitti-val", "voc-ood"],
    "speed": ["ID-bdd-OOD-coco", "OOD-open", "speed-val", "voc-ood"],
    "nu": ["ID-bdd-OOD-coco", "OOD-open", "nu-val", "voc-ood"]}
for dataset_name in dataset_list[args.id]:
    print(f"Extracting {dataset_name}")
    dataset_val = fo.load_dataset(dataset_name)
    extract(dataset_val, args.id, args.backbone)
print(f"Validation set extraction took {time.time() - t0} seconds")

#concetenate features
# print("Starting concatenating features")
# t0 = time.time()
# with open('/home/hugo/bdd100k-monitoring/train_feats/speed/resnet/speed-train_feats_tp_dict.pickle', 'rb') as f:
#     feats_dict = pickle.load(f)
# for num in range(10):
#     with open(f'/home/hugo/bdd100k-monitoring/train_feats/speed/resnet/speed-train_epoch{num+40}_feats_tp_dict.pickle', 'rb') as f:
#         data = pickle.load(f)
#     for k, v in feats_dict.items():
#         feats_dict[k] = np.concatenate((v, data[k]), axis=0)
# with open('/home/hugo/bdd100k-monitoring/train_feats/speed/resnet/speed-train_feats_tp_dict.pickle', 'wb') as f:
#     pickle.dump(feats_dict, f)
# print(f"Concatenating features took {time.time() - t0} seconds")

# #construction
print("Starting construction")
t0 = time.time()
construct(args.id, args.backbone, taus)
print(f"Construction took {time.time() - t0} seconds")

# evaluation
print("Starting evaluation")
t0 = time.time()
for tau in taus:
    evaluate(args.id, args.backbone, tau)
print(f"Evaluation took {time.time() - t0} seconds")

# tune confidence threshold
# with h5py.File(f"feats_{args.id}-train_{args.backbone}.h5", 'r') as f:
#     feats_npy = f[f'feats_{args.id}-train_{args.backbone}'][:]
# tune_threshold(dataset_train, feats_npy, args.id, args.backbone, 0.5, taus)
# monitor enhancement
# for ratio in [0.52, 0.54, 0.56, 0.58]:
#     print(f"ratio: {ratio}")
#     enlarge_evaluation(args.id, args.backbone, 1.0, ratio) 
