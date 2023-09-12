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
import re
import glob
import subprocess
import shlex

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
SPEED_THING_CLASSES = ['100kph','120kph','20kph','30kph','40kph','5kph','50kph','60kph','70kph','80kph']
NU_THING_CLASSES = ['car','truck','trailer','bus','construction_vehicle','bicycle','motorcycle','pedestrian','traffic_cone','barrier']

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
    print(accept_sum, reject_sum)
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
    elif id == "speed" or id == "prescan":
        id == "SPEED"
        eval_list = ["ID-bdd-OOD-coco", "OOD-open", "voc-ood"]
    elif id == "nu":
        id == "NU"
        eval_list = ["ID-bdd-OOD-coco", "OOD-open", "voc-ood"]
    df_summary = pd.DataFrame([[dataset_name, f"{TPR}%", f"{FPR}%"]], columns=["Dataset", "TPR", "FPR"])
    
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
        i += 1
    
    # prepare dataframes
    df_ood = pd.DataFrame(data_ood, columns=["Dataset", "FPR"])
    df_ood["Dataset"] = ["COCO", "Open Images"] if id == "voc" else ["COCO", "Open Images", "VOC-OOD"]
    print(df_summary)
    print(df_ood)
    return df_summary["TPR"].tolist()[0], df_summary["FPR"].tolist()[0], df_ood["FPR"].tolist()[0], df_ood["FPR"].tolist()[1], df_ood["FPR"].tolist()[2]

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
taus = [0.05, 0.10, 0.20, 0.50, 1.0]


# evaluation
print("Starting evaluation")
t0 = time.time()
for tau in taus:
    print(f"tau: {tau}")
    evaluate(args.id, args.backbone, tau)
print(f"Evaluation took {time.time() - t0} seconds")
