import torch
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model


import numpy as np
import argparse
import tqdm
import time
import pickle
torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
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
cfg.merge_from_file(f"resnet.yaml")
cfg.MODEL.WEIGHTS = f"model_final_{args.backbone}_{args.id}.pth" 
cfg.MODEL.DEVICE='cuda'
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(label_list)
model = build_model(cfg)
model.eval()
checkpointer = DetectionCheckpointer(model)
checkpointer.load(cfg.MODEL.WEIGHTS)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
taus = [0.05, 0.10, 0.20, 0.50, 1.0]

# construction
print("Starting construction")
t0 = time.time()
construct(args.id, args.backbone, taus)
print(f"Construction took {time.time() - t0} seconds")
