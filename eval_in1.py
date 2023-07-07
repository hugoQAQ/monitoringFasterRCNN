from pickle import load
import os
def load_monitors(id, backbone, label_list, tau):
    monitors_dict = dict()
    for class_name in label_list:
        monitor_path = f"monitors/{id}/{backbone}/{class_name}/monitor_for_clustering_parameter" + "_tau_" + str(tau) + ".pkl"
        # monitor_path = f"Monitors/{class_name}/monitor_for_clustering_parameter" + "_tau_" + str(tau) + ".pkl"
        if os.path.exists(monitor_path):
            with open(monitor_path, 'rb') as f:
                monitor = load(f)
            monitors_dict[class_name] = monitor
        else:
            print(f"monitor for {monitor_path} not found")
    return monitors_dict

import pandas as pd
import numpy as np
import pickle
import gradio as gr
import tqdm
import argparse
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
                    # 'train',
                    'motorcycle',
                    'bicycle',
                    'traffic light',
                    'traffic sign']
benchmark_vos = {"voc": {"resnet":[47.53, 51.33], "regnet":[47.77, 48.33]}, 
                 "bdd": {"resnet":[44.27, 35.54], "regnet":[36.61, 27.24]},
                 "kitti": {"resnet":[44.27, 35.54], "regnet":[36.61, 27.24]}}
KITTI_THING_CLASSES = ["Car", "Pedestrian", "Cyclist", "Van", "Truck", "Tram"]

# def tune_parameter(id, backbone, tau, delta, num_limit, progress=gr.Progress(track_tqdm=True)):
def tune_parameter(id, backbone, tau):
    if id == 'voc':
        label_list = VOC_THING_CLASSES
    elif id == 'bdd':
        label_list = BDD_THING_CLASSES
    else:
        label_list = KITTI_THING_CLASSES

    label_dict = {i:label for i, label in enumerate(label_list)}
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
        # progress(0, desc="Starting")
        for label in tqdm.tqdm(label_list, desc="Evaluation on ID data"):    
        # for label in label_list:  
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
        else:
            id == "KITTI"
            eval_list = ["ID-bdd-OOD-coco", "OOD-open", "voc-ood"]
        df_summary = pd.DataFrame([[dataset_name, f"{TPR}%", "95%", f"{FPR}%"]], columns=["Dataset", "TPR", "TPR(benchmark)", "FPR"])
        
        data_ood = []
        i = 0
        for dataset_name in tqdm.tqdm(eval_list, desc="Evaluation on OOD data"):
        # for dataset_name in eval_list: 
            accept_sum = {"tp": 0, "fp": 0}
            reject_sum = {"tp": 0, "fp": 0}
            with open(f'val_feats/{id}/{backbone}/{dataset_name}_feats_fp_dict.pickle', 'rb') as f:
                feats_fp_dict = pickle.load(f)
            for label in label_list:
                verdict = monitors_dict[label].make_verdicts(feats_fp_dict[label])
                # data_ood.append([label, len(verdict), (len(verdict)-np.sum(verdict)), (len(verdict)-np.sum(verdict))/len(verdict)])
                accept_sum["fp"] += np.sum(verdict)
                reject_sum["fp"] += len(verdict) - np.sum(verdict)
            FPR =  round((accept_sum['fp'] / (reject_sum['fp'] + accept_sum['fp'])*100), 2)
            # data_ood.append([dataset_name, accept_sum['fp'], reject_sum['fp'], (reject_sum['fp'] + accept_sum['fp']), str(FPR)+"%", benchmark[i]])

            data_ood.append([dataset_name, str(FPR)+"%", str(benchmark[i])+"%" if i<len(benchmark) else "N/A" ])
            i += 1
        
        # prepare dataframes
        # df_ood = pd.DataFrame(data_ood, columns=["class", "accepted FP", "rejected FP", "Total num.", "FPR", "FPR(benchmark)"])
        df_ood = pd.DataFrame(data_ood, columns=["Dataset", "FPR", "FPR(benchmark)"])
        df_ood["Dataset"] = ["COCO", "Open Images"] if id == "voc" else ["COCO", "Open Images", "VOC-OOD"]
        print(df_summary)
        print(df_ood)

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--backbone', type=str)
parser.add_argument('--id', type=str)
parser.add_argument('--tau', type=float)
args = parser.parse_args()
# tune_parameter(f"{args.id}", f"{args.backbone}", args.tau)
config_name = f"vos_{args.backbone}"

inference_folder = f'/home/hugo/bdd100k-monitoring/vos/detection/data/{args.id.upper()}-Detection/faster-rcnn/{config_name}/random_seed_0'
# if a folder does not exist, create it
if not os.path.exists(inference_folder):
    os.makedirs(inference_folder)

if args.id == "bdd":
    dataset_dir = "/home/hugo/fiftyone/bdd100k"
elif args.id == "voc":
    dataset_dir = "/home/hugo/fiftyone/VOC_0712_converted"
elif args.id == "nu":
    dataset_dir = "/home/hugo/nuscene"
in_eval_command = f"python vos/detection/apply_net.py  --dataset-dir {dataset_dir} --test-dataset {args.id}_custom_val  --config-file {args.id.upper()}-Detection/faster-rcnn/{config_name}.yaml  --inference-config Inference/standard_nms.yaml  --random-seed 0  --image-corruption-level 0  --visualize 0"
import subprocess
import shlex

# subprocess.call(shlex.split(in_eval_command))

if args.id == "bdd" or args.id == "kitti" or args.id == "nu":
    ood_eval_command1 = f"python vos/detection/apply_net.py  --dataset-dir /home/hugo/fiftyone/coco-2017 --test-dataset coco_ood_val_bdd  --config-file {args.id.upper()}-Detection/faster-rcnn/{config_name}.yaml  --inference-config Inference/standard_nms.yaml  --random-seed 0  --image-corruption-level 0  --visualize 0"
    # subprocess.call(shlex.split(ood_eval_command1))

    ood_eval_command2 = f"python vos/detection/apply_net.py  --dataset-dir /home/hugo/fiftyone/OpenImages --test-dataset openimages_ood_val  --config-file {args.id.upper()}-Detection/faster-rcnn/{config_name}.yaml  --inference-config Inference/standard_nms.yaml  --random-seed 0  --image-corruption-level 0  --visualize 0"
    # subprocess.call(shlex.split(ood_eval_command2))

    ood_eval_command3 = f"python vos/detection/apply_net.py  --dataset-dir /home/hugo/fiftyone/voc-ood --test-dataset voc_ood_val  --config-file {args.id.upper()}-Detection/faster-rcnn/{config_name}.yaml  --inference-config Inference/standard_nms.yaml  --random-seed 0  --image-corruption-level 0  --visualize 0"
    subprocess.call(shlex.split(ood_eval_command3))
else:
    ood_eval_command1 = f"python vos/detection/apply_net.py  --dataset-dir /home/hugo/fiftyone/coco-2017 --test-dataset coco_ood_val  --config-file {args.id.upper()}-Detection/faster-rcnn/{config_name}.yaml  --inference-config Inference/standard_nms.yaml  --random-seed 0  --image-corruption-level 0  --visualize 0"
    subprocess.call(shlex.split(ood_eval_command1))

#     ood_eval_command2 = f"python vos/detection/apply_net.py  --dataset-dir /home/hugo/fiftyone/OpenImages --test-dataset openimages_ood_val  --config-file {args.id.upper()}-Detection/faster-rcnn/{config_name}.yaml  --inference-config Inference/standard_nms.yaml  --random-seed 0  --image-corruption-level 0  --visualize 0"
#     subprocess.call(shlex.split(ood_eval_command2))

import re
import glob

file_path = f'{inference_folder}/inference/{args.id}_custom_val/standard_nms/corruption_level_0/probabilistic_scoring_res_odd*.txt'
file_path = glob.glob(file_path)[0]
threshold = re.findall(r'\d+\.\d+', file_path)[-1]

if args.id == "bdd" or args.id == "kitti" or args.id == "nu":
    metric_command1 = f"python vos/bdd_coco_plot.py --name {config_name}  --thres {threshold}  --energy 1  --seed 0 --eval_dataset coco_ood_val_bdd --id {args.id}"
    subprocess.call(shlex.split(metric_command1))

    metric_command2 = f"python vos/bdd_coco_plot.py --name {config_name}  --thres {threshold}  --energy 1  --seed 0 --eval_dataset openimages_ood_val --id {args.id}"
    subprocess.call(shlex.split(metric_command2))

    metric_command3 = f"python vos/bdd_coco_plot.py --name {config_name}  --thres {threshold}  --energy 1  --seed 0 --eval_dataset voc_ood_val --id {args.id}"
    subprocess.call(shlex.split(metric_command3))
else:
    metric_command1 = f"python vos/bdd_coco_plot.py --name {config_name}  --thres {threshold}  --energy 1  --seed 0 --eval_dataset coco_ood_val --id {args.id}"
    subprocess.call(shlex.split(metric_command1))

    metric_command2 = f"python vos/bdd_coco_plot.py --name {config_name}  --thres {threshold}  --energy 1  --seed 0 --eval_dataset openimages_ood_val --id {args.id}"
    subprocess.call(shlex.split(metric_command2))