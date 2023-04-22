import io
import platform
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (LOGGER, cv2, xyxy2xywh)
from utils.general import (Profile, non_max_suppression)
from utils.plots import Annotator, colors, save_one_box

view_img = True
save_crop = True
save_txt = True
save_img = True
save_conf = True

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


def inference_single_image(weights_path, im, view_img=True,
                           save_crop=True,
                           save_txt=True,
                           save_img=True,
                           save_conf=True):
    org_im = im
    im = np.moveaxis(np.array(im), 2, 0)
    im = np.array([im])
    weights = torch.load(weights_path, weights_only=False)

    model = DetectMultiBackend(weights_path)
    stride, names, pt = model.stride, model.names, model.pt

    # model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    dataset = LoadImages(source, img_size=512, stride=stride, auto=pt)
    with dt[0]:
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

    # Inference
    with dt[1]:
        pred = model.model(im, augment=True, visualize=True)

    # NMS
    with dt[2]:
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, False, max_det=max_det)
    print("prediction: ", pred)
    # im_file = request.files['image']
    # model = attempt_load(weights, device=torch.device('cpu'), inplace=True, fuse=False)
    save_dir = 'detections/predicted_img'
    # Process predictions
    dataset = dataset.__iter__()
    path, im, im0s, vid_cap, s = dataset.__next__()

    for i, det in enumerate(pred):  # per image
        seen += 1
        p, im, frame = Path(path), im, getattr(dataset, 'frame', 0)  # to Path
        save_path = str(f"{save_dir}/det_{p.name}")  # im.jpg
        txt_path = str(f"{save_dir}/labels/{p.stem}") + (f'_{frame}')  # im.txt
        s += '%gx%g ' % im.shape[-2:]  # print string
        gn = torch.tensor(im.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        imc = im.copy() if save_crop else im  # for save_crop
        annotator = Annotator(org_im, line_width=line_thickness, example=str(names))
        if len(det):
            # Rescale boxes from img_size to im0 size
            # det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im.shape).round()
            s = ''
            # Print results
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                if save_txt:  # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(f'{txt_path}.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                if save_crop:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
                if save_crop:
                    save_one_box(xyxy, org_im, file=Path(f'{save_dir}/crops/{names[c]}/{p.stem}.jpg'), BGR=True)

        # Stream results
        im0 = annotator.result()
        if view_img:
            if platform.system() == 'Linux' and p not in windows:
                windows.append(p)
                cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
            cv2.imshow(str(p), im0)
            cv2.waitKey(10000)  # 1 millisecond

        # Save results (image with detections)
        if save_img:
            cv2.imwrite(save_path, im0)

    # Print time (inference-only)
    LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")


if __name__ == '__main__':
    source = "detections/my_images/20230420_103714.jpg"
    im_file = open(source, "rb")
    im_bytes = im_file.read()
    im = np.array(Image.open(io.BytesIO(im_bytes)).resize((512, 512)).rotate(270))
    print(im.shape)
    # (0.227555556, 0.170666667)
    # im = F.interpolate(torch.Tensor(im),  size=(512, 512, 3))
    print(im.shape)

    print(im.shape)
    weights_path = 'exp38/weights/best.pt'
    inference_single_image(weights_path,im)
