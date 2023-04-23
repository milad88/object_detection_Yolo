import numpy as np
import torch

from models.common import DetectMultiBackend
from utils.general import (Profile, non_max_suppression)
from utils.general import (cv2)
from utils.plots import Annotator, colors

conf_thres = .250  # confidence threshold
iou_thres = 0.45  # NMS IOU threshold
max_det = 1000

nosave = False  # do not save images/videos
classes = None  # filter by class: --class 0  or --class 0 2 3
agnostic_nms = False  # class-agnostic NMS
augment = True  # augmented inference
visualize = False  # visualize features
update = False  # update all models
project = 'detections/'  # save results to project/name
name = 'exp'  # save results to project/name
exist_ok = False  # existing project/name ok  do not increment
line_thickness = 1  # bounding box thickness (pixels)
hide_labels = False  # hide labels
hide_conf = False  # hide confidences
half = False  # use FP16 half-precision inference
dnn = False  # use OpenCV DNN for ONNX inference
vid_stride = 1


def detect_frame(model, im, names):
    org_im = im

    im = np.moveaxis(np.array(im), 2, 0)
    im = np.array([im])

    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    im = torch.from_numpy(im).to(model.device)
    im = im.half() if model.fp16 else im.float()
    im /= 255.0
    pred = model.model(im, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, False, max_det=max_det)
    s = ''
    for i, det in enumerate(pred):  # per image
        seen += 1
        s += '%gx%g ' % im.shape[-2:]
        annotator = Annotator(org_im, line_width=line_thickness, example=str(names))
        if len(det):
            # Rescale boxes from img_size to im0 size
            # det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im.shape).round()
            s = ''
            # Print results
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')

                annotator.box_label(xyxy, label, color=colors(c, True))
            print(f"{s}{'' if len(det) else '(no detections), '}{dt[1].t * 1E3:.1f}ms")

        # Stream results
        res = annotator.result()
        return res


weights_path = 'exp38/weights/best.pt'
model = DetectMultiBackend(weights_path)
cap = cv2.VideoCapture(0)
names = model.names
# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

dim = (512, 512)

# resize image
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    frame = detect_frame(model, frame, names)

    cv2.imshow('Input', frame)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
