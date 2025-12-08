import os
import csv
from ultralytics import YOLO



def run_yolo_tracking(
    weights_path="finetuned-detrac-weights.pt",
    video_source="video.mp4",
    base_dir="data/processed/video1",
    csv_filename="detections_track.csv",
    run_name="yolo_track_video1",
    conf=0.5,
    iou=0.6,
    tracker="botsort.yaml",
):


    os.makedirs(base_dir, exist_ok=True)
    csv_path = os.path.join(base_dir, csv_filename)


    model = YOLO(weights_path)

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "frame_idx",     
            "track_id",       
            "class_id",
            "class_name",
            "conf",
            "x1", "y1",       
            "x2", "y2",       
            "cx", "cy_bottom",
            "bbox_w",         
            "bbox_h"          
        ])

  
        results = model.track(
            source=video_source,
            save=True,
            conf=conf,
            iou=iou,
            persist=True,
            tracker=tracker,
            agnostic_nms=True,
            stream=True,
            project=base_dir,     
            name=run_name,          
            exist_ok=True          
        )


        for frame_idx, r in enumerate(results):

            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue

            xyxy = boxes.xyxy.cpu().numpy()     
            confs = boxes.conf.cpu().numpy()   
            clss  = boxes.cls.cpu().numpy()     
            ids   = boxes.id


            if ids is None:
                track_ids = [None] * len(xyxy)
            else:
                track_ids = ids.int().cpu().numpy()


            for box, conf_val, cls_id, track_id in zip(xyxy, confs, clss, track_ids):

                x1, y1, x2, y2 = box.astype(float)


                cx = (x1 + x2) / 2.0
                cy_bottom = y2
                bbox_w = x2 - x1
                bbox_h = y2 - y1
                class_name = model.names.get(int(cls_id), str(int(cls_id)))
                track_id = int(track_id) if track_id is not None else -1

                writer.writerow([
                    frame_idx,
                    track_id,
                    int(cls_id),
                    class_name,
                    float(conf_val),
                    x1, y1,
                    x2, y2,
                    cx, cy_bottom,
                    bbox_w, bbox_h
                ])

    video_dir = os.path.abspath(os.path.join(base_dir, run_name))
    print(f"\nCSV guardado en: {os.path.abspath(csv_path)}")
    print("Video con boxes e IDs guardado en:", video_dir)

    return os.path.abspath(csv_path), video_dir
