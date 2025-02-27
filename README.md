## Installation

#### Clone the Repository

```bash
git clone https://github.com/yourusername/yolov5-crowdhuman-onnx.git
cd yolov5-crowdhuman-onnx
```

#### Install Required Packages

```bash
pip install -r requirements.txt
```

## Usage

Before running inference, you need to download weights of the YOLOv5m model trained on CrowdHuman dataset in ONNX format.

#### Download weights from the following links

**Note:** The weights are saved in FP32.

| Model Name | ONNX Model Link                                                                                           | Number of Parameters | Model Size |
| ---------- | --------------------------------------------------------------------------------------------------------- | -------------------- | ---------- |
| YOLOv5m    | [crowdhuman.onnx](https://github.com/yakhyo/yolov5-crowdhuman-onnx/releases/download/v0.0.1/crowdhuman.onnx) | 21.2M                | 84 MB      |

<br>

> If you have custom weights, you can convert your weights to ONNX format. Follow the instructions in the [YOLOv5 repository](https://github.com/ultralytics/yolov5) to convert your model. You can use the converted ONNX model with this repository.

#### Inference

```bash
python main.py --weights weights/crowdhuman.onnx --source assets/vid_input.mp4 # video
                                                 --source 0 --view # webcam and display
                                                 --source assets/img_input.jpg # image
```

- To save results add the `--save` argument and results will be saved under the `runs` folder
- To display video add the `--view` argument

**Command Line Arguments**

```
usage: main.py [-h] [--weights WEIGHTS] [--source SOURCE] [--img-size IMG_SIZE [IMG_SIZE ...]] [--conf-thres CONF_THRES] [--iou-thres IOU_THRES]
               [--max-det MAX_DET] [--save] [--view] [--project PROJECT] [--name NAME]

options:
  -h, --help            show this help message and exit
  --weights WEIGHTS     model path
  --source SOURCE       Path to video/image/webcam
  --img-size IMG_SIZE [IMG_SIZE ...]
                        inference size h,w
  --conf-thres CONF_THRES
                        confidence threshold
  --iou-thres IOU_THRES
                        NMS IoU threshold
  --max-det MAX_DET     maximum detections per image
  --save                Save detected images
  --view                View inferenced images
  --project PROJECT     save results to project/name
  --name NAME           save results to project/name
```

## Reference

1. https://github.com/ultralytics/yolov5
2. Thanks for the model weight to [SibiAkkash](https://github.com/SibiAkkash/yolov5-crowdhuman)