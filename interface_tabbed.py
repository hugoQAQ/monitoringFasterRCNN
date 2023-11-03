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
import time
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
from detectron2monitor import Detectron2Monitor


    
def fx_gradio(id, backbone, progress=gr.Progress(track_tqdm=True)):
    detectron2monitor = Detectron2Monitor(id, backbone)
    t0 = time.time()
    detectron2monitor._extract(f"{detectron2monitor.id}-train")
    minutes, seconds = divmod(time.time()-t0, 60)
    return f"Total feature extraction time: {int(minutes):02d}:{int(seconds):02d}"

def construct_gradio(id, backbone, clustering_algo, nb_clusters, eps, min_samples, progress=gr.Progress(track_tqdm=True)):
    detection2monitor = Detectron2Monitor(id, backbone)
    t0 = time.time()
    detection2monitor._construct(clustering_algo, nb_clusters, eps, min_samples)
    minutes, seconds = divmod(time.time()-t0, 60)
    return f"Total monitor construction time: {int(minutes):02d}:{int(seconds):02d}"

def fx_eval_gradio(id, backbone, progress=gr.Progress(track_tqdm=True)):
    detectron2monitor = Detectron2Monitor(id, backbone)
    t0 = time.time()
    for dataset_name in tqdm.tqdm(detectron2monitor.eval_list, desc="Evaluation data preparation"):
        detectron2monitor._extract(dataset_name)
    minutes, seconds = divmod(time.time()-t0, 60)
    return f"Total evaluation data preparation time: {int(minutes):02d}:{int(seconds):02d}"

def eval_gradio(id, backbone, clustering_algo, nb_clusters, eps, min_samples, progress=gr.Progress(track_tqdm=True)):
    detectron2monitor = Detectron2Monitor(id, backbone)
    df_id, df_ood = detectron2monitor._evaluate(clustering_algo, nb_clusters, eps, min_samples)
    return df_id, df_ood

def inference_gradio(id, backbone, clustering_algo, nb_clusters, eps, min_samples, file):
    detectron2monitor = Detectron2Monitor(id, backbone, 0.5)
    monitors_dict = detectron2monitor._load_monitors(clustering_algo, nb_clusters, eps, min_samples)
    image_dict, df, df_verdict = detectron2monitor.get_output(monitors_dict, file)
    return image_dict["detection"], image_dict["verdict"], image_dict["cam"], df, df_verdict

with gr.Blocks(theme='soft') as demo:
    gr.Markdown("# Runtime Monitoring Computer Vision Models")
    gr.Markdown(
        """
This interactive demo presents an approach to monitoring neural networks-based computer vision models using box abstraction-based techniques. Our method involves abstracting features extracted from training data to construct monitors. The demo walks users through the entire process, from monitor construction to evaluation. 
 The interface is divided into several basic modules:

- **In-distribution dataset and backbone**: This module allows users to select their target model and dataset.
- **Feature extraction**: Neuron activation pattern are extracted from the model's intermediate layers using training data. These features represent the good behaviors of the model.
- **Monitor construction**: Extracted features are grouped using different clustering techniques. These clusters are then abstracted to serve as references for the monitors. 
- **Evaluation preparation**: To facilate the evalution, the features should be extracted from evaluation datasets prior to monitor evalution. 
- **Monitor Evaluation**: The effectiveness of monitors in detecting Out-of-Distribution (OoD) objects are assessed. One of our core metric is FPR 95, which represents the false positive (incorrectly detected objects) rate when the true positive rate for ID is set at 95%. 
    """
    )
    with gr.Tab("Image Classification"):
        id = gr.Radio(['MNIST', 'CIFAR-10', 'CIFAR-100', 'ImageNet-100', 'ImageNet-1K'], label="Dataset")
        backbone = gr.Radio(['LeNet-5', 'ResNet-18', 'WideResNet-28', 'ResNet-50'], label="Backbone")
        with gr.Tab("Feature extraction"):
            extract_btn = gr.Button("Extract features")
            output1 = gr.Textbox(label="Output")
        with gr.Tab("Monitor construction"):
            construct_btn = gr.Button("Monitor Construction")
            clustering_algo = gr.Dropdown(['kmeans', 'spectral', 'dbscan', 'opticals'], label="Clustering algorithm")
            with gr.Row():
                nb_clusters = gr.Number(value=5, label="Number of clusters", precision=0)
                eps = gr.Number(value=5, label="Epsilon", precision=0)
                min_samples = gr.Number(value=10, label="Minimum samples", precision=0)
            output2 = gr.Textbox(label="Output")
        with gr.Tab("Evaluation"):
            prep_btn = gr.Button("Evaluation Data Preparation")
            prep_output = gr.Textbox(label="Output")
        with gr.Tab("Evaluation results"):
            eval_btn = gr.Button("Monitor Evaluation")
            eval_id = gr.Dataframe(type="pandas", label="ID performance")
            eavl_ood = gr.Dataframe(type="pandas", label="OOD performance")
        with gr.Tab("Inference"):
            with gr.Row().style(equal_height=True):
                with gr.Column():
                    image = gr.Image(type="filepath", label="Input")
                    button = gr.Button("Infer")
                    
                with gr.Column():
                    with gr.Tab("Detection"):
                        detection = gr.Image(label="Output")
                        df = gr.Dataframe(label="Detection summary")
                    with gr.Tab("Verdict"):
                        verdict = gr.Image(label="Output")
                        df_verdict = gr.Dataframe(label="Verdict summary")
                    with gr.Tab("Explainable AI"):
                        cam = gr.Image(label="Output")
    with gr.Tab("Object Detection"):
        id = gr.Radio(['PASCAL-VOC', 'BDD100K', 'KITTI', 'Speed signs', 'NuScenes'], label="Dataset")
        backbone = gr.Radio(['regnet', 'resnet'], label="Backbone")
        with gr.Tab("Feature extraction"):
            extract_btn = gr.Button("Extract features")
            output1 = gr.Textbox(label="Output")
        with gr.Tab("Monitor construction"):
            construct_btn = gr.Button("Monitor Construction")
            clustering_algo = gr.Dropdown(['kmeans', 'spectral', 'dbscan', 'opticals'], label="Clustering algorithm")
            with gr.Row():
                nb_clusters = gr.Number(value=5, label="Number of clusters", precision=0)
                eps = gr.Number(value=5, label="Epsilon", precision=0)
                min_samples = gr.Number(value=10, label="Minimum samples", precision=0)
            output2 = gr.Textbox(label="Output")
        with gr.Tab("Evaluation preparation"):
            prep_btn = gr.Button("Evaluation Data Preparation")
            prep_output = gr.Textbox(label="Output")
        with gr.Tab("Evaluation results"):
            eval_btn = gr.Button("Monitor Evaluation")
            eval_id = gr.Dataframe(type="pandas", label="ID performance")
            eavl_ood = gr.Dataframe(type="pandas", label="OOD performance")
        with gr.Tab("Inference"):
            with gr.Row().style(equal_height=True):
                with gr.Column():
                    image = gr.Image(type="filepath", label="Input")
                    button = gr.Button("Infer")   
                with gr.Column():
                    with gr.Tab("Detection"):
                        detection = gr.Image(label="Output")
                        df = gr.Dataframe(label="Detection summary")
                    with gr.Tab("Verdict"):
                        verdict = gr.Image(label="Output")
                        df_verdict = gr.Dataframe(label="Verdict summary")
                    with gr.Tab("Explainable AI"):
                        cam = gr.Image(label="Output")
    button.click(fn=inference_gradio, inputs=[id, backbone, clustering_algo, nb_clusters, eps, min_samples, image], outputs=[detection, verdict, cam, df, df_verdict])
    extract_btn.click(fn=fx_gradio, inputs=[id, backbone], outputs=[output1])
    construct_btn.click(fn=construct_gradio, inputs=[id, backbone, clustering_algo, nb_clusters, eps, min_samples], outputs=[output2])
    prep_btn.click(fn=fx_eval_gradio, inputs=[id, backbone], outputs=[prep_output])
    eval_btn.click(fn=eval_gradio, inputs=[id, backbone, clustering_algo, nb_clusters, eps, min_samples], outputs=[eval_id, eavl_ood])
demo.queue().launch()
