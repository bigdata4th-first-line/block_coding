# 학습된 yolo 모델 가져오기

import gradio as gr
import torch
import cv2
import numpy as np
from ultralytics import YOLO

from torchvision import transforms
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms import functional as F


from values import *


MODEL = YOLO("best.pt")


def inference_image(input_img):
    input_img = cv2.resize(input_img, (640, 640))
    img = transforms.ToTensor()(input_img).unsqueeze(0)

    result = MODEL.predict(img)
    boxes = result[0].boxes
    boxesn = boxes.xyxy

    find_cls = boxes.cls.cpu().detach().numpy()
    fin_conf = boxes.conf.cpu().detach().numpy()
    labels = []

    for cls,(x1, y1, x2, y2) in zip(find_cls, boxesn):
        labels.append(f"{CLASS[int(cls)]} {x1} {y1} {x2} {y2} ")
        print(labels)

    outs = draw_bounding_boxes(torch.from_numpy(input_img.transpose(2, 0,  1)), boxesn, labels, width=5)
    outs = F.to_pil_image(outs.detach())
    return outs

# 이미지 넣는 박스 생성
with gr.Blocks(title="YOLOv8 for Object detection") as demo:
    gr.Markdown("탭을 이용해 이미지를 넣어보세요.")
    with gr.Tab("Image"):
        with gr.Row():
            image_input = gr.Image(type="numpy", label="입력 이미지")
            out_image = gr.Image(type="numpy", label="출력 이미지")

        image_button = gr.Button("이미지 입력")
    image_button.click(inference_image, image_input, out_image)
    
demo.launch(debug=True)