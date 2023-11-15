import cv2
from ultralytics import YOLO

from torchvision import transforms

from values import *



class YoloLabel():
    def __init__(self):
        self.MODEL = YOLO("./model/best_ver3.pt")


    def inference_image(self, input_img):
        input_img = cv2.resize(input_img, (640, 640))
        img = transforms.ToTensor()(input_img).unsqueeze(0)

        result = self.MODEL.predict(img)
        boxes = result[0].boxes
        boxesn = boxes.xyxy

        find_cls = boxes.cls.cpu().detach().numpy()
        fin_conf = boxes.conf.cpu().detach().numpy()
        
        labels_list = []
        
        for cls,(x1, y1, x2, y2) in zip(find_cls, boxesn):
            label_dict = {
                'label': CLASS[int(cls)],
                'x1': int(x1),
                'y1': int(y1),
                'x2': int(x2),
                'y2': int(y2),
                'cx': int((x1+x2)/2),
                'cy': int((y1+y2)/2)
            }
            labels_list.append(label_dict)
            
        sorted_label = sorted(labels_list, key=lambda x: x['cy'])
            
        return sorted_label