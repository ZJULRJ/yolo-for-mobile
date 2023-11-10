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
from utils.datasets import LoadStreams, LoadImages,letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth,640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)
while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    # color_image=cv2.imread('sample/newimg (32).jpg')
    depth_frame = frames.get_depth_frame()
    depth_data = np.asanyarray(depth_frame.get_data())
    source=color_image
    print(source.shape)
    # cv2.imshow('Prediction', source)
    # cv2.waitKey(1)
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
    # im0s=source
    #
    # img = cv2.resize(source, (384,640))
    # img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # # 3. 将图像转换为模型期望的格式，1x3x640x640
    # img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]
    #
    # # 4. 归一化图像像素值到0到1之间
    # img = img / 255.0
    # img = torch.from_numpy(img).to(device)
    # img = img.half() if half else img.float()  # uint8 to fp16/32
    # sources = [source]  # 数据源
    # imgs = [None]
    # path = sources  # path: 图片/视频的路径
    # imgs[0] = color_image
    # im0s = imgs.copy()  # img0s: 原尺寸的图片
    # img = [letterbox(x, new_shape=imgsz)[0] for x in im0s]  # img: 进行resize + pad之后的图片
    # img = np.stack(img, 0)  # 沿着0dim进行堆叠
    # img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to 3x416x416, uint8 to float32
    # img = np.ascontiguousarray(img, dtype=np.float16 if half else np.float32)
    # # 处理每一张图片的数据格式
    # img = torch.from_numpy(img).to(device)  # 将numpy转为pytorch的tensor,并转移到运算设备上计算
    # # 如果图片是3维(RGB) 就在前面添加一个维度1当中batch_size=1
    # # 因为输入网络的图片需要是4为的 [batch_size, channel, w, h]
    # if img.ndimension() == 3:
    #     img = img.unsqueeze(0)  # 在dim0位置添加维度1，[channel, w, h] -> [batch_size, channel, w, h]
    # Inference
    # Padded resize
    im0s = source
    img = letterbox(source, 640,32)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
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
            print(img.shape[2:], im0.shape)
            for *xyxy, conf, cls in reversed(det): 
                c = int(cls)  # integer class
                label = f'{names[c]} {conf:.2f}'
                print(f'({c}) {label} xmin: {xyxy[0]} ymin: {xyxy[1]} xmax: {xyxy[2]} ymax: {xyxy[3]}')
                x=int((xyxy[0]+xyxy[2])/2)
                y=int((xyxy[1]+xyxy[3])/2)
                # print(f'depth:{depth_data[y,x]}')
            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
        # im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
        for *xyxy, conf, cls in det:
            x = int((xyxy[0] + xyxy[2]) / 2)
            y = int((xyxy[1] + xyxy[3]) / 2)
            depth = depth_data[y,x]
            lbl = names[int(cls)] 
            c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
            cv2.rectangle(im0, c1, c2, (0, 255, 0), 2)
            cv2.putText(im0, lbl, (c1[0], c1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(im0, str(depth), (c2[0], c2[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Prediction', im0)
        cv2.imwrite('img1.jpg',im0)
        cv2.waitKey(1)
        # Print time (inference + NMS)
        print(f'{s}Done. ({t2 - t1:.3f}s)')

    print(f'Done. ({time.time() - t0:.3f}s)')