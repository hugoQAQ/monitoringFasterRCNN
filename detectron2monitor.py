import torch
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.config import get_cfg
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.data.detection_utils import read_image
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


import numpy as np
import cv2
import os
import argparse
import time
import h5py
import pickle
import gradio as gr
import fiftyone as fo
from fiftyone import ViewField as F
import tqdm
import matplotlib.pyplot as plt
import io
from PIL import Image
torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from vos.detection.modeling.regnet import build_regnet_fpn_backbone
import core.metadata as metadata

from utils_clustering import *

from base_cam import EigenCAM
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget

fullName2ab_dict = {'PASCAL-VOC':"voc", 'BDD100K':"bdd", 'KITTI':"kitti", 'Speed signs':"speed", 'NuScenes':"nu"}
ab2FullName_dict = {'voc':"PASCAL-VOC", 'bdd':"BDD100K", 'kitti':"KITTI", 'speed':"Speed signs", 'nu':"NuScenes"}
class Detectron2Monitor():
    def __init__(self, id, backbone, confidence_threshold=0.05):
        self.id, self.label_list = self._get_label_list(id)
        self.backbone = backbone
        self.confidence_threshold = confidence_threshold
        self.cfg, self.device, self.model = self._get_model()
        self.label_dict = {i:label for i, label in enumerate(self.label_list)}
        self.eval_list = ["ID-voc-OOD-coco", "OOD-open", "voc-val"] if self.id == "voc" else ["ID-bdd-OOD-coco", "OOD-open", "voc-ood", f"{self.id}-val"]
        MetadataCatalog.get("custom_dataset").set(thing_classes=self.label_list)
    
    def _get_label_list(self, id):
        id = fullName2ab_dict[id]
        if id == 'voc':      
            label_list = metadata.VOC_THING_CLASSES
        elif id == 'bdd':
            label_list = metadata.BDD_THING_CLASSES
        elif id == 'kitti':
            label_list = metadata.KITTI_THING_CLASSES
        elif id == 'speed' or id == 'prescan':
            label_list = metadata.SPEED_THING_CLASSES
        else:
            label_list = metadata.NU_THING_CLASSES
        return id, label_list

    def _get_model(self):
        cfg = get_cfg()
        cfg.merge_from_file(f"/home/hugo/bdd100k-monitoring/monitoringObjectDetection/vanilla_{self.backbone}.yaml")
        cfg.MODEL.WEIGHTS = f"models/model_final_{self.backbone}_{self.id}.pth" 
        cfg.MODEL.DEVICE='cuda'
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.label_list)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
        model = build_model(cfg)
        model.eval()
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        return cfg, device, model

    def _inference(self, model, inputs):
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

    def _save_features(self, feats_npy, dataset_view, file_path):
        features_idx_dict = {cls:[] for cls in self.label_list}
        for sample in tqdm.tqdm(dataset_view, desc="Saving features"):
            for detection in sample.prediction.detections:
                label_pred = detection.label
                feature_idx = detection.feature_idx
                features_idx_dict[label_pred].append(feature_idx)
        feats_dict = {cls:feats_npy[features_idx_dict[cls]] for cls in self.label_list}
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
                pickle.dump(feats_dict, f)

    def _extract(self, dataset_name):
        dataset = fo.load_dataset(dataset_name)
        aug = T.AugmentationList([T.ResizeShortestEdge(
                [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST], self.cfg.INPUT.MAX_SIZE_TEST),
            ]
        )
        i = 0
        feats_list = []
        for sample in tqdm.tqdm(dataset, desc="Extracting features"):
            image = read_image(sample.filepath, format="BGR")
            height, width = image.shape[:2]
            inputs = [self._get_input_dict(image)]
            preds, feats = self._inference(self.model, inputs)
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
                label = self.label_dict[label]
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
        with h5py.File(f'feats_{self.id}-train_{self.backbone}.h5', 'w') as f:
            dset = f.create_dataset(f"feats_{self.id}-train_{self.backbone}", data=feats_npy)
        if dataset.name.endswith(("train", "val")):
            results = dataset.evaluate_detections(
                "prediction",
                gt_field="detections",
                eval_key="eval",
                compute_mAP=True)
            # results.print_report()
            # print("mAP: ", results.mAP())
            tp_prediction_view = dataset.filter_labels("prediction", F("eval") == "tp")
            self._save_features(feats_npy, tp_prediction_view, f"train_feats/{self.id}/{self.backbone}/{self.id}-train_feats_tp_dict.pickle")  
            if dataset.name.endswith("val"):
                fp_prediction_view = dataset.filter_labels("prediction", F("eval") == "fp")
                self._save_features(feats_npy, fp_prediction_view, f"val_feats/{self.id}/{self.backbone}/{self.dataset_name}_feats_fp_dict.pickle")
        else:
            self._save_features(feats_npy, dataset, f"val_feats/{self.id}/{self.backbone}/{self.dataset_name}_feats_fp_dict.pickle")  
    
    def _construct(self, clustering_algo, nb_clusters=4, eps=5, min_samples=10):
        with open(f"/home/hugo/bdd100k-monitoring/train_feats/{self.id}/{self.backbone}/{self.id}-train_feats_tp_dict.pickle", 'rb') as f:
            feats_dict = pickle.load(f)
        dir_path = f'monitors/{self.id}/{self.backbone}/{clustering_algo}'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        monitor_dict = {}   
        for class_, fts in tqdm.tqdm(feats_dict.items(), desc="Constructing monitors"):
            if len(fts) == 0:
                continue
            if clustering_algo == "kmeans":
                clusters = k_means_cluster(fts, nb_clusters)
            elif clustering_algo == "spectral":
                clusters = spectral_cluster(fts, nb_clusters)
            elif clustering_algo == "dbscan":
                clusters = dbscan_cluster(fts, eps, min_samples)
            dims = fts.shape[1]
            box_list = []
            for cl_id, points in clusters.items():
                box = Box()
                box.build(dims, points)
                box_list.append(box)
            monitor = Monitor(good_ref=box_list)
            monitor_dict[class_] = monitor
        if clustering_algo == "dbscan":
            with open(f"monitors/{self.id}/{self.backbone}/{clustering_algo}/eps{eps}_min_samples{min_samples}.pkl" , 'wb') as f:
                pickle.dump(monitor_dict, f)
        else:
            with open(f"monitors/{self.id}/{self.backbone}/{clustering_algo}/{nb_clusters}.pkl" , 'wb') as f:
                pickle.dump(monitor_dict, f)

    def _load_monitors(self, clustering_algo, nb_clusters, eps=5, min_samples=10):
        if clustering_algo == "dbscan":
            with open(f"monitors/{self.id}/{self.backbone}/{clustering_algo}/eps{eps}_min_samples{min_samples}.pkl", 'rb') as f:
                monitors_dict = pickle.load(f)  
        else:
            with open(f"monitors/{self.id}/{self.backbone}/{clustering_algo}/{nb_clusters}.pkl", 'rb') as f:
                monitors_dict = pickle.load(f)
        return monitors_dict
    
    def _evaluate(self, clustering_algo, nb_clusters, eps, min_samples):
        dataset_name = f"{self.id}-val"
        with open(f'val_feats/{self.id}/{self.backbone}/{dataset_name}_feats_tp_dict.pickle', 'rb') as f:
            feats_tp_dict = pickle.load(f)
        with open(f'val_feats/{self.id}/{self.backbone}/{dataset_name}_feats_fp_dict.pickle', 'rb') as f:
            feats_fp_dict = pickle.load(f)
        monitors_dict = self._load_monitors(clustering_algo, nb_clusters, eps, min_samples)
        # make verdicts on ID data
        data_tp = []
        data_fp = []
        accept_sum = {"tp": 0, "fp": 0}
        reject_sum = {"tp": 0, "fp": 0}
        for label in tqdm.tqdm(self.label_list, desc="Evaluation on ID data"): 
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
        id_name = ab2FullName_dict[self.id]
        df_id = pd.DataFrame([[id_name, f"{TPR}%", f"{FPR}%"]], columns=["Dataset", "TPR", "FPR"])
        
        data_ood = []
        i = 0
        self.eval_list.remove(dataset_name)
        for dataset_name in tqdm.tqdm(self.eval_list, desc="Evaluation on OOD data"):
            accept_sum = {"tp": 0, "fp": 0}
            reject_sum = {"tp": 0, "fp": 0}
            with open(f'val_feats/{self.id}/{self.backbone}/{dataset_name}_feats_fp_dict.pickle', 'rb') as f:
                feats_fp_dict = pickle.load(f)
            for label in self.label_list:
                if label in monitors_dict:
                    verdict = monitors_dict[label].make_verdicts(feats_fp_dict[label])
                    accept_sum["fp"] += np.sum(verdict)
                    reject_sum["fp"] += len(verdict) - np.sum(verdict)
            FPR =  round((accept_sum['fp'] / (reject_sum['fp'] + accept_sum['fp'])*100), 2)
            data_ood.append([dataset_name, str(FPR)+"%"])
            i += 1
        # prepare dataframes
        df_ood = pd.DataFrame(data_ood, columns=["Dataset", "FPR"])
        df_ood["Dataset"] = ["COCO", "Open Images"] if self.id == "voc" else ["COCO", "Open Images", "VOC-OOD"]
        return df_id, df_ood

    def _postprocess_cam(self, raw_cam, img_width, img_height):
        cam_orig = np.sum(raw_cam, axis=0)  # [H,W]
        cam_orig = np.maximum(cam_orig, 0)  # ReLU
        cam_orig -= np.min(cam_orig)
        cam_orig /= np.max(cam_orig)
        cam = cv2.resize(cam_orig, (img_width, img_height))
        return cam

    def _fasterrcnn_reshape_transform(self, x):
        target_size = x['p6'].size()[-2 : ]
        activations = []
        for key, value in x.items():
            activations.append(torch.nn.functional.interpolate(torch.abs(value), target_size, mode='bilinear'))
        activations = torch.cat(activations, axis=1)
        return activations
    
    def _get_input_dict(self, original_image):
        height, width = original_image.shape[:2]
        transform_gen = T.ResizeShortestEdge(
        [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST], self.cfg.INPUT.MAX_SIZE_TEST
        )
        image = transform_gen.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = {"image": image, "height": height, "width": width}
        return inputs
    
    def get_output(self, monitors_dict, img):
        image = read_image(img, format="BGR")
        input_image_dict = [self._get_input_dict(image)]
        pred_instances, feats = self._inference(self.model, input_image_dict)
        detections = pred_instances[0]["instances"].to("cpu")
        cls_idxs = detections.pred_classes.detach().numpy()
        # get labels from class indices
        labels = [self.label_dict[i] for i in cls_idxs]
        # count values in labels, and return a dictionary
        labels_count_dict = dict((i, labels.count(i)) for i in labels)
        v = Visualizer(image[..., ::-1], MetadataCatalog.get("custom_dataset"), scale=1)
        v = v.draw_instance_predictions(detections)
        img_detection = v.get_image()
        df = pd.DataFrame(list(labels_count_dict.items()), columns=['Object', 'Count'])
        verdicts = []
        for label, feat in zip(labels, feats):
            verdict = monitors_dict[label].make_verdicts(feat[np.newaxis,:])[0]
            verdicts.append(verdict)
        detections_ood = detections[[i for i, x in enumerate(verdicts) if not x]]
        detections_ood.pred_classes = torch.tensor([5]*len(detections_ood.pred_classes))
        labels_ood = [label for label, verdict in zip(labels, verdicts) if not verdict]
        verdicts_ood = ["Rejected"]*len(labels_ood)
        df_verdict = pd.DataFrame(list(zip(labels_ood, verdicts_ood)), columns=['Object', 'Verdict'])
        v = Visualizer(image[..., ::-1], MetadataCatalog.get("custom_dataset"), scale=1)
        for box in detections_ood.pred_boxes.to('cpu'):
            v.draw_box(box)
            v.draw_text("OOD", tuple(box[:2].numpy()))
        v = v.get_output()
        img_ood = v.get_image()
        pred_bboxes = detections.pred_boxes.tensor.numpy().astype(np.int32)
        target_layers = [self.model.backbone]
        targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=pred_bboxes)]
        cam = EigenCAM(self.model,
                        target_layers, 
                        use_cuda=False,
                        reshape_transform=self._fasterrcnn_reshape_transform)
        grayscale_cam = cam(input_image_dict, targets)
        cam = self._postprocess_cam(grayscale_cam, input_image_dict[0]["width"], input_image_dict[0]["height"])
        plt.rcParams["figure.figsize"] = (30,10)
        plt.imshow(img_detection[..., ::-1], interpolation='none')
        plt.imshow(cam, cmap='jet', alpha=0.5)
        plt.axis("off")
        img_buff = io.BytesIO()
        plt.savefig(img_buff, format='png', bbox_inches='tight', pad_inches=0)
        img_cam = Image.open(img_buff)
        image_dict = {}
        image_dict["image"] = image
        image_dict["cam"] = img_cam
        image_dict["detection"] = img_detection
        image_dict["verdict"] = img_ood
        return image_dict, df, df_verdict