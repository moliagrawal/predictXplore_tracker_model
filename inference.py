import cv2
import torch
import numpy as np
import argparse
import csv
from pathlib import Path
from boxmot import BoostTrack

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default="/app/inputs/video.mp4")
    parser.add_argument("--output", default="/app/outputs/output_live_feed.mp4")
    parser.add_argument("--csv", default="/app/outputs/results.csv")
    parser.add_argument("--cfg", default="yolov3.cfg")
    parser.add_argument("--names", default="coco.names")
    parser.add_argument("--weights", default="yolov3.weights")
    parser.add_argument("--reid", default="clip_market1501.pt")
    args = parser.parse_args()

    net = cv2.dnn.readNetFromDarknet(args.cfg, args.weights)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    width, height = 416, 416
    prediction_threshold = 0.25

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    tracker = BoostTrack(
        reid_weights=Path(args.reid),
        device=device,
        half=False
    )

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))

    csv_rows = []
    frame_idx = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (width, height), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))

                objectness = detection[4]
                class_score = scores[class_id]
                confidence = float(objectness * class_score)

                if confidence > prediction_threshold and class_id == 0:
                    cx, cy, w, h = detection[0:4]
                    cx *= frame.shape[1]
                    cy *= frame.shape[0]
                    w *= frame.shape[1]
                    h *= frame.shape[0]

                    x = int(cx - w / 2)
                    y = int(cy - h / 2)

                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(confidence)
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, prediction_threshold, 0.4)

        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                x1 = x
                y1 = y
                x2 = x + w
                y2 = y + h
                detections.append([x1, y1, x2, y2, confidences[i], class_ids[i]])

        detections = np.array(detections) if len(detections) > 0 else np.empty((0, 6))

        tracks = tracker.update(detections, frame)

        # Save CSV rows
        if tracks is not None and len(tracks) > 0:
            for t in tracks:
                x1, y1, x2, y2 = map(int, t[:4])
                track_id = int(t[4]) if len(t) > 4 else -1
                conf = float(t[5]) if len(t) > 5 else 0.0
                csv_rows.append([frame_idx, track_id, x1, y1, x2, y2, conf])

        tracker.plot_results(frame, show_trajectories=True)
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    with open(args.csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "track_id", "x1", "y1", "x2", "y2", "confidence"])
        writer.writerows(csv_rows)

if __name__ == "__main__":
    main()