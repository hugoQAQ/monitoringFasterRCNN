# Box abstraction based runtime monitor for FastarRCNN model
This is the source code accompanying the paper xxx
## Requirements
```
pip install -r requiremnets.txt
```
To install Detectron2, please follow [here](https://github.com/facebookresearch/detectron2).
## Dataset Preparation
We use [Fiftyone](https://docs.voxel51.com) library to load and visualize datasets. 

BDD100k, COCO, KITTI and OpenImage can be loaded directly through [Fiftyone Datasets Zoo](https://docs.voxel51.com/user_guide/dataset_zoo/datasets.html?highlight=zoo).

For other datasets, such as NuScene can be loaded manually via the following simple pattern:
```python
import fiftyone as fo

# A name for the dataset
name = "my-dataset"

# The directory containing the dataset to import
dataset_dir = "/path/to/dataset"

# The type of the dataset being imported
dataset_type = fo.types.COCODetectionDataset  # for example

dataset = fo.Dataset.from_dir(
    dataset_dir=dataset_dir,
    dataset_type=dataset_type,
    name=name,
)
```
The custom dataset folder should have the following structure:
```
 └── /path/to/dataset
     |
     ├── Data
     └── labels.json
```
Notice that the annotation file `labels.json` should be prepared in COCO format.

The OOD datasets' json files can be downloaded in [COCO-OOD](to-add), [Open-OOD](to-add) and [VOC-OOD](to-add).
## Evaluation
**Pretrained models**

We provide the weights and the evaluation metrics for both Resnet and RegNet backbones.
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Dataset</th>

<th valign="bottom">ResNet</th>
<!-- TABLE BODY -->
<!-- ROW: BDD -->
<th valign="bottom">RegNet</th>
<!-- TABLE BODY -->
<!-- ROW: BDD -->
 <tr><td align="left">BDD100k</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/137257644/model_final_721ade.pkl">weight</a>&nbsp;|&nbsp;<a href="bdd-metrics.txt">metrics</a></td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/137257644/model_final_721ade.pkl">weight</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/137257644/metrics.json">metrics</a></td>
</tr>
 <tr><td align="left">KITTI</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/137257644/model_final_721ade.pkl">weight</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/137257644/metrics.json">metrics</a></td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/137257644/model_final_721ade.pkl">weight</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/137257644/metrics.json">metrics</a></td>
</tr>
 <tr><td align="left">NuScene</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/137257644/model_final_721ade.pkl">weight</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/137257644/metrics.json">metrics</a></td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/137257644/model_final_721ade.pkl">weight</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/137257644/metrics.json">metrics</a></td>
</tr>
</tbody></table>
