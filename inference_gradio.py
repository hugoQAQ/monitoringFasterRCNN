# %%
import torch

import detectron2.data.transforms as T
from detectron2.utils.visualizer import Visualizer
from detectron2.data.detection_utils import read_image
from detectron2.data import MetadataCatalog
from detectron2.structures import Instances, Boxes

import numpy as np
import cv2
from PIL import Image
import random
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import io
from pickle import load
import glob
import onnxruntime as ort
from fiftyone import ViewField as F
torch.manual_seed(10)
np.random.seed(10)
random.seed(10)

from grad_cam.eigen_cam import EigenCAM
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
def setup_model(model_path):
        ort_session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
        input_name = ort_session.get_inputs()[0].name
        return ort_session, input_name
SPEED_THING_CLASSES = ['100kph','120kph','20kph','30kph','40kph','5kph','50kph','60kph','70kph','80kph']
label_list = SPEED_THING_CLASSES
id = "prescan"
backbone = "resnet"
def load_monitor(tau):
    monitors_dict = {}
    for class_name in label_list:
        if class_name == "train" or class_name == "OOD":
            continue
        monitor_path = f"monitors/{id}/{backbone}/{class_name}/monitor_for_clustering_parameter" + "_tau_" + str(tau) + ".pkl"
        with open(monitor_path, 'rb') as f:
            monitor = load(f)
        monitors_dict[class_name] = monitor
        delta = 0.5
        for i in range(len(monitors_dict[class_name].good_ref)):
            monitors_dict[class_name].good_ref[i].ivals = monitors_dict[class_name].good_ref[i].ivals*np.array([1-delta, 1+delta])
    return monitors_dict
ort_session, input_name = setup_model("model.onnx")
monitors_dict = load_monitor(tau=1.0)
class Detectron2Monitor():
    def __init__(self, label_list, label_dict, tau, threshold):
        self.label_list = label_list
        self.class_dict = label_dict
    
    def _get_input_tensor(self, image):
        transform_gen = T.Resize((800, 1200))
        image_trans = transform_gen.get_transform(image).apply_image(image)
        image = torch.as_tensor(image_trans.astype("float32").transpose(2, 0, 1))
        return image_trans, image

    def get_output(self, img):
        image = read_image(img, format="BGR")
        image_original, input_image = self._get_input_tensor(image)
        
        outputs = ort_session.run(None, {input_name: input_image.numpy()})
        boxes = Boxes(outputs[2])
        image_size = outputs[-1]
        prediction = Instances(image_size)
        prediction.pred_boxes = boxes
        prediction.pred_classes = torch.tensor(outputs[3])
        prediction.scores = torch.tensor(outputs[-2])
        preds = [{"instances":prediction}]
        detections = preds[0]["instances"].to("cpu")
        cls_idxs = detections.pred_classes.detach().numpy()
        scores = detections.scores.detach().numpy()
        labels = [label_dict[i] for i in cls_idxs]
        # count values in labels, and return a dictionary
        v = Visualizer(image_original[..., ::-1], MetadataCatalog.get("demo_dataset"), scale=1)
        v = v.draw_instance_predictions(detections)
        img_detection = v.get_image()
        df = pd.DataFrame(columns=['Object', 'Confidence'])
        df['Object'] = labels
        df['Confidence'] = scores
        feats = outputs[0]
        verdicts = []
        for label, feat in zip(labels, feats):
            verdict = monitors_dict[label].make_verdicts(feat[np.newaxis,:])[0]
            verdicts.append(verdict)
        detections_ood = detections[[i for i, x in enumerate(verdicts) if not x]]
        detections_ood.pred_classes = torch.tensor([10]*len(detections_ood.pred_classes))
        verdicts_ood = ["Accepted" if verdict else "Rejected" for verdict in verdicts]
        df_verdict = pd.DataFrame(list(zip(labels, verdicts_ood)), columns=['Object', 'Verdict'])
        v = Visualizer(image_original[..., ::-1], MetadataCatalog.get("demo_dataset"), scale=1)
        for box in detections_ood.pred_boxes.to('cpu'):
            v.draw_box(box)
            v.draw_text("OOD", tuple(box[:2].numpy()))
        v = v.get_output()
        img_ood = v.get_image()
        image_dict = {}
        image_dict["image"] = image
        image_dict["detection"] = img_detection
        image_dict["verdict"] = img_ood
        return image_dict, df, df_verdict
id = "prescan"
backbone = "resnet"
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
if id == 'voc':
        label_list = VOC_THING_CLASSES
elif id == 'bdd':
    label_list = BDD_THING_CLASSES
elif id == 'kitti':
    label_list = KITTI_THING_CLASSES
elif id == 'speed' or id == 'prescan':
    label_list = SPEED_THING_CLASSES
label_list.append("OOD")
label_dict = {i:label for i, label in enumerate(label_list)}
MetadataCatalog.get("demo_dataset").set(thing_classes=label_list)

def inference_gd(file, threshold):
    extractor = Detectron2Monitor(label_list=label_list, label_dict=label_dict, tau=1.0, threshold=threshold)
    image_dict, df, df_verdict = extractor.get_output(file)
    return image_dict["detection"], image_dict["verdict"], df, df_verdict




# examples = ["examples/0.jpg", "examples/1.jpg", "examples/2.jpg", "examples/3.jpg"]

examples = list(glob.glob("examples-speed/*"))
with gr.Blocks(theme="gradio/monochrome") as demo:
    gr.Markdown("# Runtime Monitoring Object Detection")
#     gr.Markdown(
#         """This interactive demo is based on the box abstraction-based monitors for Faster R-CNN model. The model is trained using [Detectron2](https://github.com/facebookresearch/detectron2) library on the in-distribution dataset [Berkeley DeepDrive-100k](https://www.bdd100k.com/), which contains objects within autonomous driving domain. The monitors are constructed by abstraction of extracted feature from the training data. The demo showcases the monitors' capacity to reject problematic detections due to out-of-distribution(OOD) objects. 

# To utilize the demo, upload an image or select one from the provided examples, and click on "Infer" to view the following results: 

# - **Detection**: outputs of Object Detector
# - **Detection summary**: a summary of the detection outputs
# - **Verdict**: verdicts from Monitors
# - **Explainable AI**: visual explanation generated by [grad-cam](https://github.com/jacobgil/pytorch-grad-cam) library which is based on Class Activation Mapping(CAM) method.

# In case the output image seems too small, simply right-click on the image, and choose “Open image in new tab” to visualize it in full size.
#     """
#     )
    with gr.Row():
        with gr.Column():
            image = gr.inputs.Image(type="filepath", label="Input")
            score_threshold = gr.inputs.Slider(minimum=0.05, maximum=1.0, step=0.05, default=0.5, label="Confidence score threshold")
            button = gr.Button("Infer")
            eamples_block = gr.Examples(examples, image)
        with gr.Column():
            with gr.Tab("Detection"):
                detection = gr.Image(label="Output")
                df = gr.Dataframe(label="Detection summary")
            with gr.Tab("Verdict"):
                verdict = gr.Image(label="Output")
                df_verdict = gr.Dataframe(label="Verdict summary")
            
    button.click(fn=inference_gd, inputs=[image, score_threshold], outputs=[detection, verdict, df, df_verdict])

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



