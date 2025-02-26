import os
import cv2
import logging
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import time
from typing import List, Tuple

from models import YOLOv5
from utils.general import check_img_size, scale_boxes, draw_detections, colors, increment_path, LoadMedia

from deep_sort_realtime.deepsort_tracker import DeepSort



def run_object_detection(
    weights: str,
    source: str,
    img_size: List[int],
    conf_thres: float,
    iou_thres: float,
    max_det: int,
    save: bool,
    view: bool,
    project: str,
    name: str
):
    if save:
        save_dir = increment_path(Path(project) / name)
        save_dir.mkdir(parents=True, exist_ok=True)


    model = YOLOv5(weights, conf_thres, iou_thres, max_det)
    img_size = check_img_size(img_size, s=model.stride)
    dataset = LoadMedia(source, img_size=img_size)

    tracker = DeepSort(max_age=6, n_init=3, nms_max_overlap=0.5)

    up_line = (720,(560,710))
    down_line = (710,(700,880))

    previous_positions = {}
    up_count = 0
    down_count = 0

    counts_by_interval = defaultdict(lambda: {"up": 0, "down": 0})
    interval_duration = 5  


    # For writing video and webcam
    vid_writer = None
    if save and dataset.type in ["video", "webcam"]:
        cap = dataset.cap
        save_path = str(save_dir / os.path.basename(source))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))


    model_performance = 0
    tracker_performance = 0
    iteration = 0

    for resized_image, original_image, status, second in dataset:
        # Model Inference
        iteration += 1


        start = time.perf_counter()

        boxes, scores, class_ids = model(resized_image)

        end = time.perf_counter()


        model_performance += end - start 


        # Scale bounding boxes to original image size
        boxes = scale_boxes(resized_image.shape, boxes, original_image.shape).round()

        start = time.perf_counter()

        detections = []
        for box, score, class_id in zip(boxes, scores, class_ids):
            label = model.names[int(class_id)]
            if label == "person":
                bbox_tlwh = xyxy_to_tlwh(np.array([box]))[0]  
                detections.append((bbox_tlwh.tolist(), score, int(class_id))) 

        tracked_objects = tracker.update_tracks(detections, frame=original_image)

        interval_key = (int(second) // interval_duration) * interval_duration

        for track in tracked_objects:
            if not track.is_confirmed():
                continue  
    
            track_id = track.track_id  
            ltrb = track.to_ltrb()  
            
            x_center = (ltrb[0] + ltrb[2]) / 2
            y_center = (ltrb[1] + ltrb[3]) / 2
            
            class_id = track.det_class

            label = f"ID {track_id} - {model.names[int(class_id)]}"

            if track_id in previous_positions:
                
                x , y = previous_positions[track_id]

                if up_line[1][0] <= y <= up_line[1][1] and up_line[1][0] <= y_center <= up_line[1][1] and x_center > up_line[0] and x < up_line[0]:
                    up_count += 1
                    counts_by_interval[interval_key]["up"] += 1
                    print(f"Objektum {track_id} felment a mozgolepcsovel!")
                
                if down_line[1][0] <= y <= down_line[1][1] and down_line[1][0] <= y_center <= down_line[1][1] and x_center < down_line[0] and x > down_line[0]:
                    down_count += 1
                    counts_by_interval[interval_key]["down"] += 1
                    print(f"Objektum {track_id} lement a mozgolepcsovel!")

            previous_positions[track_id] = (x_center, y_center)





            draw_detections(original_image, ltrb, 1.0, label, colors(int(class_id)))

        end = time.perf_counter()

        tracker_performance += end - start

        print(status)

        if view:
            # Display the image with detections
            cv2.imshow('Webcam Inference', original_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break

        if save:
            if dataset.type == "image":
                save_path = str(save_dir / f"frame_{dataset.frame:04d}.jpg")
                cv2.imwrite(save_path, original_image)
            elif dataset.type in ["video", "webcam"]:
                vid_writer.write(original_image)

    if save and vid_writer is not None:
        vid_writer.release()

    if save:
        print(f"Results saved to {save_dir}")


    print(f"Felmentek: {up_count} | Valódi: 9")
    print(f"Lementek: {down_count}| Valódi: 6")

    print(f"Model átlag frame feldolgozés: {model_performance / iteration:.4f} seconds")
    print(f"Tracker átlag futás idő: {tracker_performance / iteration:.4f} seconds")

    cv2.destroyAllWindows()

    plot_intervals(counts_by_interval, interval_duration)


def plot_intervals(counts_by_interval, interval_duration):
    interval_keys = sorted(counts_by_interval.keys())
    up_counts = [counts_by_interval[k]["up"] for k in interval_keys]
    down_counts = [counts_by_interval[k]["down"] for k in interval_keys]

    x_labels = [f"{k}-{k+interval_duration}" for k in interval_keys]
    x_indexes = np.arange(len(x_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x_indexes - width/2, up_counts, width, label='Felment')
    ax.bar(x_indexes + width/2, down_counts, width, label='Lement')

    ax.set_xlabel('Időintervallumok (másodperc)')
    ax.set_ylabel('Események száma')
    ax.set_title('Fel- és lemozgások a mozgólépcsőn 5 másodperces intervallumokra')
    ax.set_xticks(x_indexes)
    ax.set_xticklabels(x_labels)
    ax.legend()

    plt.tight_layout()
    plt.show()

def xyxy_to_tlwh(bbox_xyxy):
    tlwh = bbox_xyxy.copy()
    tlwh[:, 2] = tlwh[:, 2] - tlwh[:, 0]  # width
    tlwh[:, 3] = tlwh[:, 3] - tlwh[:, 1]  # height
    return tlwh

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="weights\crowdhuman.onnx", help="model path")
    parser.add_argument("--source", type=str, default="assets\input.mp4", help="Path to video/image/webcam")
    parser.add_argument("--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.40, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--save", action="store_true", help="Save detected images")
    parser.add_argument("--view", action="store_true", help="View inferenced images")
    parser.add_argument("--project", default="runs", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    args = parser.parse_args()
    args.img_size = args.img_size * 2 if len(args.img_size) == 1 else args.img_size
    return args

def main():
    params = parse_args()
    run_object_detection(
        weights=params.weights,
        source=params.source,
        img_size=params.img_size,
        conf_thres=params.conf_thres,
        iou_thres=params.iou_thres,
        max_det=params.max_det,
        save=params.save,
        view=params.view,
        project=params.project,
        name=params.name
    )


if __name__ == "__main__":
    main()
