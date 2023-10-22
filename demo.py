import argparse
import time
from pathlib import Path
import pyrealsense2 as rs
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
while True:
    pipeline = rs.pipeline()
    config = rs.config() 
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start(config)
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())

    source=color_image
    cv2.imshow('Prediction', source)
    cv2.waitKey(1)
    weights='weights/best (1).pt'
    imgsz = 640
    # Initialize
    set_logging()
    device = select_device('cpu')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    im0s=source
    img = cv2.resize(source, (640,640))

    # 3. 将图像转换为模型期望的格式，1x3x640x640
    img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]

    # 4. 归一化图像像素值到0到1之间
    img = img / 255.0
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, 0.45, 0.5, classes=None, agnostic=1)
    t2 = time_synchronized()

    # Apply Classifier
    if classify:
        pred = apply_classifier(pred, modelc, img, im0s)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        s, im0 = '', im0s
        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det): 
                c = int(cls)  # integer class
                label = f'{names[c]} {conf:.2f}'
                print(f'({c}) {label} xmin: {xyxy[0]} ymin: {xyxy[1]} xmax: {xyxy[2]} ymax: {xyxy[3]}')
            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
        im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB) 
        for *xyxy, conf, cls in det:
            lbl = names[int(cls)] 
            c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
            cv2.rectangle(im0, c1, c2, (0, 255, 0), 2)
            cv2.putText(im0, lbl, (c1[0], c1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Prediction', im0)
        cv2.waitKey(1)
        # Print time (inference + NMS)
        print(f'{s}Done. ({t2 - t1:.3f}s)')

    print(f'Done. ({time.time() - t0:.3f}s)')