# %%
import torch
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.config import get_cfg
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.utils.visualizer import Visualizer
from detectron2.data.detection_utils import read_image
from detectron2.data import MetadataCatalog

import numpy as np
import cv2
from PIL import Image
import random
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import io
from pickle import load

from fiftyone import ViewField as F
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

from grad_cam.eigen_cam import EigenCAM
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget

class Detectron2Monitor():
    def __init__(self, label_list, label_dict, config_file, model_file, tau):
        self.label_list = label_list
        self.cfg = self._setup_cfg(config_file, model_file)
        self.model = build_model(self.cfg)
        self.model.eval()
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(self.cfg.MODEL.WEIGHTS)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_dict = label_dict
        self.monitors_dict = self._load_monitor(tau=tau)

        
    def _setup_cfg(self, config_file, model_file):  
        cfg = get_cfg()
        cfg.merge_from_file(config_file)
        cfg.MODEL.WEIGHTS = model_file
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        if torch.cuda.is_available():
            cfg.MODEL.DEVICE = "cuda"
        else:
            cfg.MODEL.DEVICE = "cpu"
        cfg.freeze()
        return cfg
    
    def _get_input_dict(self, original_image):
        height, width = original_image.shape[:2]
        transform_gen = T.ResizeShortestEdge(
        [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST], self.cfg.INPUT.MAX_SIZE_TEST
        )
        image = transform_gen.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = {"image": image, "height": height, "width": width}
        return inputs
    
    def _postprocess_cam(self, raw_cam, img_width, img_height):
        cam_orig = np.sum(raw_cam, axis=0)  # [H,W]
        cam_orig = np.maximum(cam_orig, 0)  # ReLU
        cam_orig -= np.min(cam_orig)
        cam_orig /= np.max(cam_orig)
        cam = cv2.resize(cam_orig, (img_width, img_height))
        return cam

    def _inference(self, inputs):
        with torch.no_grad():
            images = self.model.preprocess_image(inputs)  
            features = self.model.backbone(images.tensor)  
            proposals, _ = self.model.proposal_generator(images, features, None)  # RPN

            features_ = [features[f] for f in self.model.roi_heads.box_in_features]
            box_features = self.model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])
            box_features = self.model.roi_heads.box_head(box_features)  # features of all 1k candidates
            predictions = self.model.roi_heads.box_predictor(box_features)
            pred_instances, pred_inds = self.model.roi_heads.box_predictor.inference(predictions, proposals)
            pred_instances = self.model.roi_heads.forward_with_given_boxes(features, pred_instances)

            # output boxes, masks, scores, etc
            pred_instances = self.model._postprocess(pred_instances, inputs, images.image_sizes)  # scale box to orig size
            # features of the proposed boxes
            feats = box_features[pred_inds]
        return pred_instances, feats 
     
    def _load_monitor(self, tau):
        monitors_dict = {}
        for class_name in self.label_list:
            if class_name == "train" or class_name == "OOD":
                continue
            monitor_path = f"monitors/{class_name}/monitor_for_clustering_parameter" + "_tau_" + str(tau) + ".pkl"
            with open(monitor_path, 'rb') as f:
                monitor = load(f)
            monitors_dict[class_name] = monitor
        return monitors_dict
    
    def _fasterrcnn_reshape_transform(self, x):
        target_size = x['p6'].size()[-2 : ]
        activations = []
        for key, value in x.items():
            activations.append(torch.nn.functional.interpolate(torch.abs(value), target_size, mode='bilinear'))
        activations = torch.cat(activations, axis=1)
        return activations

    def get_output(self, img):
        image = read_image(img, format="BGR")
        input_image_dict = [self._get_input_dict(image)]
        
        pred_instances, feats = self._inference(input_image_dict)
        feats = feats.cpu().detach().numpy()
        detections = pred_instances[0]["instances"].to("cpu")
        cls_idxs = detections.pred_classes.detach().numpy()
        # get labels from class indices
        labels = [self.class_dict[i] for i in cls_idxs]
        # count values in labels, and return a dictionary
        labels_count_dict = dict((i, labels.count(i)) for i in labels)
        v = Visualizer(image[..., ::-1], MetadataCatalog.get("bdd_dataset"), scale=1)
        v = v.draw_instance_predictions(detections)
        img_detection = v.get_image()
        df = pd.DataFrame(list(labels_count_dict.items()), columns=['Object', 'Count'])
        verdicts = []
        for label, feat in zip(labels, feats):
            verdict = self.monitors_dict[label].make_verdicts(feat[np.newaxis,:])[0] if label in self.monitors_dict else True
            verdicts.append(verdict)
        detections_ood = detections[[i for i, x in enumerate(verdicts) if not x]]
        detections_ood.pred_classes = torch.tensor([10]*len(detections_ood.pred_classes))
        v = Visualizer(image[..., ::-1], MetadataCatalog.get("bdd_dataset"), scale=1)
        v = v.draw_instance_predictions(detections_ood)
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
        return image_dict, df

config_file = "vos/detection/configs/BDD-Detection/faster-rcnn/vanilla.yaml"
model_file = "model_final_vos_resnet_bdd.pth"
label_dict = {
            0: 'pedestrian',
            1: 'rider',
            2: 'car',
            3: 'truck',
            4: 'bus',
            5: 'train',
            6: 'motor',
            7: 'bike',
            8: 'traffic light',
            9: 'traffic sign',
            10: 'OOD'
        }
label_list = list(label_dict.values())
MetadataCatalog.get("bdd_dataset").set(thing_classes=label_list)
extractor = Detectron2Monitor(config_file=config_file, label_list=label_list, label_dict=label_dict, model_file=model_file, tau=0.05)

# %%
def inference_gd(file):
    image_dict, df = extractor.get_output(file)
    return image_dict["detection"], df, image_dict["verdict"], image_dict["cam"]


examples = ["examples/0.jpg", "examples/1.jpg", "examples/2.jpg", "examples/3.jpg"]
with gr.Blocks(theme="gradio/monochrome") as demo:
    gr.Markdown("# Runtime Monitoring Object Detection")
    gr.Markdown(
        """This interactive demo is based on the box abstraction-based monitors for Faster R-CNN model. The model is trained using [Detectron2](https://github.com/facebookresearch/detectron2) library on the in-distribution dataset [Berkeley DeepDrive-100k](https://www.bdd100k.com/), which contains objects within autonomous driving domain. The monitors are constructed by abstraction of extracted feature from the training data. The demo showcases the monitors' capacity to reject problematic detections due to out-of-distribution(OOD) objects. 

To utilize the demo, upload an image or select one from the provided examples, and click on "Infer" to view the following results: 

- **Detection**: outputs of Object Detector
- **Detection summary**: a summary of the detection outputs
- **Verdict**: verdicts from Monitors
- **Explainable AI**: visual explanation generated by [grad-cam](https://github.com/jacobgil/pytorch-grad-cam) library which is based on Class Activation Mapping(CAM) method.

In case the output image seems too small, simply right-click on the image, and choose “Open image in new tab” to visualize it in full size.
    """
    )
    with gr.Row():
        with gr.Column():
            image = gr.inputs.Image(type="filepath", label="Input")
            button = gr.Button("Infer")
            eamples_block = gr.Examples(examples, image)
        with gr.Column():
            with gr.Tab("Detection"):
                detection = gr.Image(label="Output")
            with gr.Tab("Verdict"):
                verdict = gr.Image(label="Output")
            with gr.Tab("Explainable AI"):
                cam = gr.Image(label="Output")
            df = gr.Dataframe(label="Detection summary")
    button.click(fn=inference_gd, inputs=image, outputs=[detection, df, verdict, cam])

demo.launch()
# gradio_interface = gr.Interface(
# inference_gd,
# inputs=gr.inputs.Image(type="filepath"),
# outputs=[gr.Image(label="Detection"), gr.Dataframe(label="Nb. of detected objects"), gr.Image(label="Verdict"), gr.Image(label="Explainable AI")],
# title="Runtime Monitoring Object Detection",
# description="Box abstraction-based monitors for Faster R-CNN algorithm",
# examples=["examples/0.jpg", "examples/1.jpg", "examples/2.jpg", "examples/3.jpg", "examples/4.jpg"])

# # Launch the interface
# gradio_interface.launch()

# %%



