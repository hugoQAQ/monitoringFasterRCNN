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
import argparse

# create a parser object
parser = argparse.ArgumentParser(description='Description of your program')

# add an optional "--model" argument 
parser.add_argument('--id', type=str, help='Description of the in-distribution dataset argument')
parser.add_argument('--tau', nargs='+', type=float, help='Description of the tau argument')
parser.add_argument('--backbone', type=str, help='Description of the backbone argument')
args = parser.parse_args()
id = args.id
taus = args.tau
backbone = args.backbone
label_list = VOC_THING_CLASSES if id == 'voc' else BDD_THING_CLASSES
label_dict = {i:label for i, label in enumerate(label_list)}

# %%
from util import *
for label in label_list:
    t0 = time.time()
    feature_path = f"feats_{label}.npy"
    # feature_path = f"Bdd_feats_20k/feats_{label}.npy"
    features_npy = np.load(feature_path)
    print(f"{label} feature number: {features_npy.shape[0]}")
    if features_npy.shape[0] == 0:
        continue
    clustering_results  = features_clustering(features_npy, taus)
    monitor_saving_folder = f"monitors/{id}/{backbone}/{label}/"
    # monitor_saving_folder = f"Monitors/{label}/"
    monitor_construction_from_features(features_npy, taus, clustering_results, label, monitor_saving_folder)
    print(f"Class {label}: {time.time()-t0} s" )

import os
import glob
npy_files = glob.glob('*.npy')
for npy_file in npy_files:
    os.remove(npy_file)




