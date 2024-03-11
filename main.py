# load YOLOv8
from ultralytics import YOLO
import cv2

# load video
model = YOLO('yolov8n.pt')

# read video frames
video_path = './highway.mp4'
cap = cv2.VideoCapture(video_path)

ret = True

while ret:
    ret, frame = cap.read()

    # tracking
    results = model.track(frame, persist=True)

    # bounding
    cur_frame = results[0].plot()
    cv2.imshow('frame', cur_frame)
    if cv2.waitKey(10) & 0xFF == ord('x'):
        break

