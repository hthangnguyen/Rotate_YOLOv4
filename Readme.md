[Paper for Rotate YOLOv4](https://github.com/hthangnguyen/Rotate_YOLOv4/blob/main/paper_eng_preprint.pdf)
# INSTALLATION
## Requirements
* Python 3.7+
* Pytorch ≥ 1.7
* CUDA 10.0 or higher

I have tested on:
* OS: Windows 10 and Ubuntu 18.04
* CUDA: 10.2/11.0

### Dataset

**I tested the model on 3 datasets**

[UCAS-High Resolution Aerial Object Detection Dataset (UCAS-AOD)](https://github.com/ming71/UCAS-AOD-benchmark)</br>
[Vehicle Detection in Aerial Imagery (VEDAI)(512)](https://downloads.greyc.fr/vedai/)</br>
[DLR Multi-class Vehicle Detection and Orientation in Aerial Imagery (DLR-MVDA)](https://www.dlr.de/eoc/en/desktopdefault.aspx/tabid-12760/22294_read-52777)</br>

**Oriented bounding box label format:**
```
obj_id    ncx    ncy    nw    nh    theta

* ncx, ncy: normalized center coordinate
* nw, nh: normalized width and height (nw ≥ nh)
* theta: radian angle between w and x-axis [-pi/2, pi/2)

```

#### Loss Function (only for x, y, w, h, theta)

<img src="https://i.imgur.com/zdA9RJj.png" alt="loss" height="90"/>
<img src="https://i.imgur.com/Qi1XFXS.png" alt="angle" height="70"/>



### Usage

Make sure your files arrangment looks like the following
```
R-YOLOv4/
├── train.py
├── test.py
├── detect.py
├── requirements.txt
├── model/
├── tools/
├── outputs/
├── weights
    ├── pretrained/ (for training)
    └── dataset_name(UCAS/VEDAI/DLR)/ (for testing and detection)
└── data
    ├── coco.names
    ├── train
        ├── images
            ├── ...png/jpg
            └── ...txt
    ├── test
        ├── images
            ├── ...png/jpg
            └── ...txt
    └── detect
        └── ...png/jpg
```

### Train

```
usage: train.py [-h] [--data_folder DATA_FOLDER] [--weights_path WEIGHTS_PATH] [--model_name MODEL_NAME] [--epochs EPOCHS] [--lr LR]
                [--batch_size BATCH_SIZE] [--subdivisions SUBDIVISIONS] [--img_size IMG_SIZE] [--number_of_classes NUMBER_OF_CLASSES]
                [--no_augmentation] [--no_multiscale]
```

### Test

```
usage: test.py [-h] [--data_folder DATA_FOLDER] [--model_name MODEL_NAME] [--class_path CLASS_PATH] [--conf_thres CONF_THRES]
               [--nms_thres NMS_THRES] [--iou_thres IOU_THRES] [--batch_size BATCH_SIZE] [--img_size IMG_SIZE]
               [--number_of_classes NUMBER_OF_CLASSES]
```

### Detect

```
usage: detect.py [-h] [--data_folder DATA_FOLDER] [--model_name MODEL_NAME] [--class_path CLASS_PATH] [--conf_thres CONF_THRES]
                 [--nms_thres NMS_THRES] [--batch_size BATCH_SIZE] [--img_size IMG_SIZE] [--number_of_classes NUMBER_OF_CLASSES]
```

<img src="https://github.com/hthangnguyen/Rotate_YOLOv4/blob/main/outputs/vedai_detect.png" alt="vedai" height="430"/>
<img src="https://github.com/hthangnguyen/Rotate_YOLOv4/blob/main/outputs/dlr_detect.jpg" alt="dlr" height="430"/>



## Ackknowledgements
I have used utility functions from other open-source projects.

[Tianxiaomo/pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4)</br>
[R-YOLOv4](https://github.com/kunnnnethan/R-YOLOv4)</br>
[YOLOv5_OBB](https://github.com/hukaixuan19970627/yolov5_obb)</br>
